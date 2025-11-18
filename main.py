import torch
import glob
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
import pickle
import utils
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from transformers import GPT2LMHeadModel, GPT2Config, TransfoXLLMHeadModel, TransfoXLConfig
from saver import Saver
import subprocess

## set the input length. must be same with the model config
X_LEN = 1024

def parse_opt():
    parser = argparse.ArgumentParser()
    ####################################################
    # you can define your arguments here. there is a example below.
    # parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    ####################################################
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='basic_event_dictionary.pkl')
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    parser.add_argument('--ckp_folder', type=str, help='the checkpoint folder path.', required=True)
    parser.add_argument('--model_type', type=str, help='model type: gpt2 or transformer_xl.', default='gpt2')
    parser.add_argument('--overwrite', action='store_true', help='whether overwrite the ckp folder.')
    parser.add_argument('--lr', action='store', type=float, default=0.0002, help='learning rate.')
    parser.add_argument('--inference', action='store_true', help='inference only.')
    parser.add_argument('--epoch_to_inference', type=int, help='the epoch number of the model to inference.', default=200)
    parser.add_argument('--inference_output_path', type=str, help='the output path for inference.', default='./results/')
    parser.add_argument('--num_inference_files', type=int, help='number of inference files.', default=20)
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
vocab_size = len(event2word)
print(f'Vocabulary size: {vocab_size}')


class MIDIDataset(Dataset):
    def __init__(self, midi_l = [], prompt = ''):
        self.midi_l = midi_l
        self.x_len = X_LEN
        self.dictionary_path = opt.dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)

    def __len__(self):
        return len(self.parser)  
    
    def __getitem__(self, index):
        return self.parser[index]
    
    def chord_extract(self, midi_path, max_time):
        ####################################################
        # add your chord extraction method here if you want
        ####################################################
        return
    
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end

        # if you use chord items, you need to add chord_items into "items"
        # e.g. items = tempo_items + note_items + chord_items
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths, return_words=False):
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)

        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        # all_words is a list containing words list of all midi files
        # all_words = [[tokens of midi], [tokens of midi], ...]
        if return_words:
            return all_words

        # you can cut the data into what you want to feed into model
        # Warning : this example cannot use in transformer_XL, you must implement group segments by yourself
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            # abandon last segments in a midi
            pairs = pairs[0:len(pairs)-(len(pairs)%5)]
            segments = segments + pairs
        segments = np.array(segments)
        # print(segments.shape)
        return segments

class Model(nn.Module):
    def __init__(self, model_type='gpt2'):
        super(Model, self).__init__()
        #################################################
        # TODO: create your model here
        # Support GPT2 and Transformer-XL
        #################################################
        self.model_type = model_type
        if model_type == 'gpt2':
            print('Using GPT2 model')
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=1024,
                n_embd=768,
                n_layer=12,
                n_head=12
            )
            self.model = GPT2LMHeadModel(config)
        elif model_type == 'transformer_xl':
            print('Using Transformer-XL model')
            config = TransfoXLConfig(
                vocab_size=vocab_size,
                d_model=512,
                n_layer=12,
                n_head=8,
                cutoffs=[]
            )
            self.model = TransfoXLLMHeadModel(config)
        # parameters amount
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total model parameters: {total_params}')
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable model parameters: {trainable_params}')

    def forward(self, x, attention_mask=None):
        #################################################
        # create your model here
        #################################################
        if self.model_type == 'gpt2':
            return self.model(x, attention_mask=attention_mask).logits
        elif self.model_type == 'transformer_xl':
            return self.model(input_ids=x).logits

def temperature_sampling(logits, temperature, topk, top_p):
    # CRUCIAL: Sanitize logits to prevent NaNs/Infs from scrambling indices
    logits = np.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)    
    # temperature scaling
    logits = logits / temperature
    
    # Step 1: Get top-k logits and their ORIGINAL indices
    sorted_indices = np.argsort(logits)[::-1]  # Descending order
    topk_indices = sorted_indices[:topk]
    topk_logits = logits[topk_indices]
    
    # Step 2: Convert to probabilities (softmax on top-k)
    topk_probs = np.exp(topk_logits - np.max(topk_logits))  # Numerical stability
    topk_probs = topk_probs / topk_probs.sum()
    
    # Step 3: Apply top-p (nucleus) filtering
    sorted_prob_indices = np.argsort(topk_probs)[::-1]  # Sort probabilities descending
    sorted_probs = topk_probs[sorted_prob_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find cutoff: keep tokens until cumulative prob exceeds top_p
    cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1  # +1 to include the token that crosses threshold
    cutoff_idx = max(1, cutoff_idx)  # Keep at least 1 token
    cutoff_idx = min(cutoff_idx, len(sorted_probs))  # Don't exceed array bounds
    
    # Step 4: Get final indices (map back to ORIGINAL vocab indices)
    nucleus_prob_indices = sorted_prob_indices[:cutoff_idx]
    final_vocab_indices = topk_indices[nucleus_prob_indices]
    final_probs = topk_probs[nucleus_prob_indices]
    
    # Renormalize probabilities
    final_probs = final_probs / final_probs.sum()
    
    # Step 5: Sample from the final distribution
    word = np.random.choice(final_vocab_indices, p=final_probs)
    return int(word)  # Ensure it's a Python int, not numpy type

def test(model_type='gpt_2', n_target_bar = 32, temperature = 1.2, topk = 5, top_p = 0.9, output_path = '', model_path = '', prompt_file=None):
    # check path folder
    try:
        os.makedirs('./results', exist_ok=True)
        print("dir \'./results\' is created")
    except:
        pass

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    with torch.no_grad():
        # load model
        checkpoint = torch.load(model_path, weights_only=False)
        model = Model(model_type=model_type).to(opt.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        batch_size = 1

        if prompt_file is not None:
            # Use provided prompt file
            words = MIDIDataset(midi_l=[]).prepare_data([prompt_file], return_words=True)
            # print(words)
        else:  
            # Or, random select prompt to start
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                if 'chord' in opt.dict_path:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        print('Start generating')
        while current_generated_bar < n_target_bar:
            print("\r", current_generated_bar, end="")
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])
            
            # FIX: Prevent sequence from exceeding max position embeddings (1024)
            # Take only the last 1024 tokens if sequence grows too long
            if temp_x.shape[1] > 1024:
                temp_x = temp_x[:, -1024:]
            
            temp_x = torch.Tensor(temp_x).long()
            
            
            output_logits = model(temp_x.to(opt.device))
            
            # sampling
            _logit = output_logits[0, -1].to('cpu').detach().numpy()
            word = temperature_sampling(
                logits=_logit, 
                temperature=temperature,
                topk=topk,
                top_p=top_p 
            )
            
            # Bounds checking: ensure sampled token is valid
            vocab_size = len(event2word)
            if word >= vocab_size or word < 0:
                print(f"\n!!! ERROR: Sampled word {word} is out of bounds [0, {vocab_size-1}]")
                print(f"Clamping to valid range...")
                word = max(0, min(word, vocab_size - 1))

            words[0].append(word)

            if word == event2word['Bar_None']:
                current_generated_bar += 1
        
        utils.write_midi(
            words=words[0],
            word2event=word2event,
            output_path=output_path,
            prompt_path=None)

def render_midi_to_wav(midi_path, output_wav_path, soundfont_path="TimGM.sf2", sample_rate=44100):
    midi_path = os.path.abspath(midi_path)
    output_wav_path = os.path.abspath(output_wav_path)
    soundfont_path = os.path.abspath(soundfont_path)
    
    subprocess.run([
        "fluidsynth",
        "-F", output_wav_path,
        "-r", str(sample_rate),
        soundfont_path,
        midi_path
    ], check=True)
    return output_wav_path

    
def train(is_continue = False, checkpoints_path = '', model_type='gpt2'):
    
    epochs = 200

    # create data list
    # use glob to get all midi file path
    assert os.path.exists('Pop1K7/midi_analyzed/'), "Please download and extract the Pop1K7 dataset into the current folder."
    train_list = glob.glob('Pop1K7/midi_analyzed/**/*.mid')
    print('train list len =', len(train_list))

    # dataset
    train_dataset = MIDIDataset(train_list)
    # dataloader
    BATCH_SIZE = 4
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    print('Dataloader is created')

    # saver
    saver = Saver(save_dir=opt.ckp_folder, val_acc=False)
    os.makedirs(os.path.join(opt.ckp_folder, 'checkpoints'), exist_ok=True)

    print('Device:', opt.device)
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device(opt.device)
    else:
        device = torch.device("cpu")
    
    # create model
    if not is_continue:
        start_epoch = 1
        model = Model(model_type=model_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1

        model = Model().to(device)
        model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Model is created \nStart training')
    
    model.train()
    losses = []
    try:
        os.makedirs(opt.ckp_folder)
        print("dir is created")
    except:
        pass
    
    for epoch in range(start_epoch, epochs+1):
        single_epoch = []
        for i in tqdm(train_dataloader):
            # x, y shape = (batch_size, length)
            x = i[:, 0, :].to(device).long()
            y = i[:, 1, :].to(device).long()
            output_logit = model(x)
            loss = nn.CrossEntropyLoss()(output_logit.permute(0, 2, 1), y)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            single_epoch.append(loss.to('cpu').mean().item())
            optimizer.step()
            optimizer.zero_grad()
        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch,losses[-1]))
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': losses[-1],
                    }, os.path.join(opt.ckp_folder, 'checkpoints', 'epoch_%03d.pkl'%epoch))
        np.save(os.path.join(opt.ckp_folder, 'training_loss'), np.array(losses))
        saver.saver_epoch(losses[-1])

def main(model_type='gpt2'):
    ######################################
    # write your main function here
    ######################################
    train(is_continue=False, model_type=model_type)
    return

if __name__ == '__main__':
    model_type = opt.model_type # 'gpt2' or 'transformer_xl'
    output_dir = opt.inference_output_path
    print(f'ckpt dir: {opt.ckp_folder}')
    if not opt.inference:
        if opt.overwrite:   # for safety of existing results
            os.makedirs(opt.ckp_folder, exist_ok=True)
        else:
            os.makedirs(opt.ckp_folder)
        main(model_type=model_type)
    
    for i in tqdm(range(opt.num_inference_files)):
        os.makedirs(output_dir, exist_ok=True)
        midi_path = os.path.join(output_dir, f'generated_{i}.mid')
        wav_path = os.path.join(output_dir, f'generated_{i}.wav')
        print(f'Generating file {i}... store to {midi_path} and {wav_path}')
        # generate midi
        test(
            model_type=model_type,
            n_target_bar = 32,
            temperature = 1.2,
            topk = 5,
            top_p = 0.9,
            output_path = midi_path,
            model_path = os.path.join(opt.ckp_folder, 'checkpoints', f'epoch_{opt.epoch_to_inference}.pkl'),
            prompt_file=None
        )
        # render to wav
        render_midi_to_wav(
            midi_path=midi_path,
            output_wav_path=wav_path,
            soundfont_path="TimGM.sf2",
            sample_rate=44100
        )
        print(f'Generated {midi_path} and {wav_path}')

    # Continuation file
    prompt_files = glob.glob('prompt_song/*.mid')
    os.makedirs(os.path.join(output_dir, 'continuation'), exist_ok=True)
    for prompt_file in prompt_files:
        midi_path = os.path.join(output_dir, f'continuation/{os.path.basename(prompt_file)}')
        wav_path = os.path.join(output_dir, f'continuation/{os.path.splitext(os.path.basename(prompt_file))[0]}.wav')
        # generate midi
        test(
            model_type=model_type,
            n_target_bar = 24,
            temperature = 1.2,
            topk = 5,
            top_p = 0.9,
            output_path = midi_path,
            model_path = os.path.join(opt.ckp_folder, 'checkpoints', f'epoch_{opt.epoch_to_inference}.pkl'),
            prompt_file=prompt_file
        )
        # render to wav
        render_midi_to_wav(
            midi_path=midi_path,
            output_wav_path=wav_path,
            soundfont_path="TimGM.sf2",
            sample_rate=44100
        )
        print(f'Generated continuation {midi_path} and {wav_path}')