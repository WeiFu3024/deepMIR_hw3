import os, shutil
import numpy as np
import torch
import re
from miditok import REMI, CPWord, TokenizerConfig
from src.model import Model
from src.dataset import MIDIDataset
from src.utils import *
from saver import Saver


def compute_loss(outputs, labels):
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous() # drop last predicted token (no ground truth for it)
    shift_labels = labels[..., 1:].contiguous() # drop first token (always predict from second token)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def train(args, tokenizer):
    dataset = MIDIDataset(base_dir=args.data.base_dir, pkl_dir=args.data.pkl_dir, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.env.num_workers,
        prefetch_factor=args.env.prefetch_factor,
        pin_memory=True
    )
    
    # Get vocabulary size from tokenizer
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    model = Model(model_type=args.model.type, vocab_size=vocab_size).to(args.env.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
    lr_scheduler = setup_lr_scheduler(optimizer, args, len(dataloader))
    saver = Saver(save_dir=args.save_dir, val_acc=False)
    
    model.train()
    for epoch in range(args.train.epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch.to(args.env.device)
            
            # Create labels for language modeling (shifted input)
            labels = input_ids.clone()
            outputs = model(input_ids)
            loss = compute_loss(outputs, labels)
            
            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
            # Logging
            if (batch_idx + 1) % args.train.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.train.epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                saver.saver_batch({
                    'epoch': epoch + 1,
                    'step': batch_idx + 1 + epoch * len(dataloader),
                    'lr': lr_scheduler.get_last_lr()[0],
                    'loss': loss.item(),
                })
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.train.epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")
        saver.saver_epoch(avg_epoch_loss)
        saver.save_model(model, f"model_epoch_{epoch+1}.pth")
    print("Training finished.")

if __name__ == "__main__":
    config_path = "src/config.yaml"
    args = load_config(config_path)
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(args.save_dir, 'config.yaml'))
    shutil.copy(__file__, os.path.join(args.save_dir, 'train.py'))

    set_seed(args.env.seed)
    
    beat_res_str = args.data.tokenizer.beat_res 
    beat_res_converted = {}
    for key_str, value in beat_res_str.items():
        # Use regular expression to find numbers inside the string key
        numbers = re.findall(r'\d+', key_str)
        key_tuple = tuple(map(int, numbers))
        beat_res_converted[key_tuple] = value
    

    tokenizer_params = {
        "pitch_range": args.data.tokenizer.pitch_range,
        "beat_res": beat_res_converted,
        "num_velocities": args.data.tokenizer.num_velocities,
        "special_tokens": args.data.tokenizer.special_tokens,
        "use_chords": args.data.tokenizer.use_chords,
        "use_rests": args.data.tokenizer.use_rests,
        "use_tempo": args.data.tokenizer.use_tempo,
        "use_time_signatures": args.data.tokenizer.use_time_signatures,
        "use_programs": args.data.tokenizer.use_programs,
        "tempo_range": args.data.tokenizer.tempo_range,
    }
    tokenizer_config = TokenizerConfig(**tokenizer_params)

    if args.data.midi_rep == 'REMI':
        tokenizer = REMI(tokenizer_config)
    elif args.data.midi_rep == 'CPWord':
        tokenizer = CPWord(tokenizer_config)
    else:
        raise ValueError(f"Unknown MIDI representation: {args.data.midi_rep}")
    
    train(args, tokenizer)