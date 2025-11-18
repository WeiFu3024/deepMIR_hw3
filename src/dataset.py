import os
from torch.utils.data import Dataset
import numpy as np
from symusic import Score
from miditok import REMI, CPWord, TokenizerConfig
import pickle
from tqdm import tqdm


def get_file_list(base_dir, extension=".mid"):
    file_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list


class MIDIDataset(Dataset):
    def __init__(self, 
                 base_dir, 
                 pkl_dir,
                 tokenizer, 
                 x_len=1024):
        self.base_dir = base_dir
        self.file_list = get_file_list(base_dir, extension=".mid")
        self.pkl_dir = pkl_dir
        self.tokenizer = tokenizer
        self.x_len = x_len
        self.metadata = self.make_metadata()

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        midi_name, start_idx = self.metadata[idx]
        pkl_path = os.path.join(self.pkl_dir, midi_name + '.pkl')
        with open(pkl_path, 'rb') as f:
            token_ids = pickle.load(f)
        seq = token_ids[start_idx:start_idx + self.x_len] if start_idx + self.x_len <= len(token_ids)\
            else token_ids[start_idx:] + [0] * (self.x_len - (len(token_ids) - start_idx))  # pad if needed
        return np.array(seq, dtype=np.int64)

    def make_metadata(self):
        # make filewise pkl files for tokenized data, also create metadata for indexing
        print("Start Preprocessing MIDI files and creating metadata...")
        metadata = []
        os.makedirs(self.pkl_dir, exist_ok=True)
        for midi_path in tqdm(self.file_list, desc="Processing MIDI files"):
            base_name = os.path.basename(midi_path)
            pkl_path = os.path.join(self.pkl_dir, base_name + '.pkl')
            
            if os.path.exists(pkl_path):
                # Load existing tokenized data to create metadata
                with open(pkl_path, 'rb') as f:
                    ids = pickle.load(f)
                metadata.extend([(base_name, i) for i in range(0, len(ids), self.x_len)])
            else:
                score = Score(midi_path)
                tok_seqs = self.tokenizer(score)
                ids = [tid for seq in tok_seqs for tid in seq.ids]
                # create metadata entries
                metadata.extend([(os.path.basename(midi_path), i) for i in range(0, len(ids), self.x_len)])
                # save tokenized data as pkl
                with open(pkl_path, 'wb') as f:
                    pickle.dump(ids, f)
            
        print(f"Metadata created! Total metadata entries: {len(metadata)}")
        return metadata


if __name__ == "__main__":
    tokenizer_params = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": True,
        "use_rests": True,
        "use_tempo": True,
        "use_time_signatures": False,
        "use_programs": False,
        "tempo_range": (40, 250),
    }
    tokenizer_config = TokenizerConfig(**tokenizer_params)
    remi_tokenizer = REMI(tokenizer_config)
    print(f"Vocabulary size: {len(remi_tokenizer)}")
    cp_tokenizer = CPWord(tokenizer_config)
    print(f"Vocabulary size (CPWord): {len(cp_tokenizer)}")
    dataset = MIDIDataset(base_dir="Pop1K7/midi_analyzed", pkl_dir="Pop1K7/midi_analyzed/remi_pkl", tokenizer=remi_tokenizer)
    print(f"Number of MIDI files: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample tokens from first MIDI file: {sample[:5]}...; Total Length: {len(sample)}")  # Print first 5 tokens
    dataset_cp = MIDIDataset(base_dir="Pop1K7/midi_analyzed", pkl_dir="Pop1K7/midi_analyzed/cp_pkl", tokenizer=cp_tokenizer)
    sample_cp = dataset_cp[0]
    print(f"Compound Word Rep Length: {len(sample_cp[0])}")
    print(f"Sample tokens from first MIDI file (CP format): {sample_cp[:5]}...; Total Length: {len(sample_cp)}")  # Print first 5 tokens

