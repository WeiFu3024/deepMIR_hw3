# Since the GPT2 loss curve is lost, we re-inference to calculate losses here, note that this is 
# not the same as the training loss curve, but only an approximation, since the loss plot in training is average over epoch, during optimization steps

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from main import Model, MIDIDataset
from src.utils import load_config, setup_lr_scheduler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import glob
import os
import pickle
from tqdm import tqdm
from saver import Saver

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
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
vocab_size = len(event2word)
print(f'Vocabulary size: {vocab_size}')

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = Model(model_type='gpt2')

    assert os.path.exists('Pop1K7/midi_analyzed/'), "Please download and extract the Pop1K7 dataset into the current folder."
    train_list = glob.glob('Pop1K7/midi_analyzed/**/*.mid')
    print('train list len =', len(train_list))

    # dataset
    dataset = MIDIDataset(train_list)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    saver = Saver(save_dir=opt.ckp_folder, val_acc=False)
    device = opt.device
    
    for epoch in range(200):
        model = load_model(model, os.path.join(opt.ckp_folder, 'checkpoints', f'epoch_{(i+1):03}.pkl'), opt.device)
        print(f'Calculating loss for epoch {i}...')

        total_loss = 0.0
        for i in tqdm(dataloader):
            input_ids = i.to(opt.device)
            x = i[:, 0, :].to(device).long()
            y = i[:, 1, :].to(device).long()
            output_logit = model(x)
            loss = nn.CrossEntropyLoss()(output_logit.permute(0, 2, 1), y)
            loss.backward()
            total_loss += loss.item()
        
        avg_epoch_loss = total_loss / len(dataloader)
        saver.saver_epoch(avg_epoch_loss)
        print(f'Epoch {epoch} loss: {avg_epoch_loss}')
