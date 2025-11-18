import sys
import os
import math
import time
import pickle
import json
import numpy as np
import argparse

import torch
import torch.nn as nn

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

# Import from main-cp.py
from main_cp import TransformerModel, sampling, write_midi

################################################################################
# Constants
################################################################################
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

################################################################################
# MIDI to Representation Conversion
################################################################################

def read_midi_to_events(path_midi, word2event, event2word):
    """
    Read a MIDI file and convert it to the compound word representation
    Returns: numpy array of shape (seq_len, 7)
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(path_midi)
    
    # Extract tempo changes
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append({
            'time': tempo.time,
            'tempo': tempo.tempo
        })
    
    # Extract chord markers
    chord_items = []
    for marker in midi_obj.markers:
        chord_items.append({
            'time': marker.time,
            'chord': marker.text
        })
    
    # Extract notes
    note_items = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note_items.append({
                    'start': note.start,
                    'end': note.end,
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })
    
    # Sort notes by start time
    note_items.sort(key=lambda x: x['start'])
    
    # Convert to compound words
    events = []
    current_bar = 0
    current_tempo = None
    current_chord = None
    
    # Group notes by position
    positions = {}
    for note in note_items:
        start_pos = note['start']
        if start_pos not in positions:
            positions[start_pos] = []
        positions[start_pos].append(note)
    
    sorted_positions = sorted(positions.keys())
    
    for pos in sorted_positions:
        # Calculate bar and beat
        bar = pos // BAR_RESOL
        beat = (pos % BAR_RESOL) // TICK_RESOL
        
        # Add bar event if new bar
        if bar > current_bar:
            events.append([
                event2word['tempo'].get('CONTI', 0),
                event2word['chord'].get('CONTI', 0),
                event2word['bar-beat']['Bar'],
                event2word['type']['Metrical'],
                event2word['pitch'].get('CONTI', 0),
                event2word['duration'].get('CONTI', 0),
                event2word['velocity'].get('CONTI', 0)
            ])
            current_bar = bar
        
        # Find tempo at this position
        tempo_val = None
        for tempo_item in tempo_items:
            if tempo_item['time'] <= pos:
                tempo_val = tempo_item['tempo']
        
        # Find chord at this position
        chord_val = None
        for chord_item in chord_items:
            if chord_item['time'] <= pos:
                chord_val = chord_item['chord']
        
        # Add beat event
        tempo_str = f'Tempo_{int(tempo_val)}' if tempo_val else 'CONTI'
        chord_str = chord_val if chord_val else 'CONTI'
        beat_str = f'Beat_{beat}'
        
        events.append([
            event2word['tempo'].get(tempo_str, 0),
            event2word['chord'].get(chord_str, 0),
            event2word['bar-beat'][beat_str],
            event2word['type']['Metrical'],
            event2word['pitch'].get('CONTI', 0),
            event2word['duration'].get('CONTI', 0),
            event2word['velocity'].get('CONTI', 0)
        ])
        
        # Add note events for this position
        for note in positions[pos]:
            pitch_str = f'Pitch_{note["pitch"]}'
            duration = note['end'] - note['start']
            duration_str = f'Duration_{duration}'
            velocity_str = f'Velocity_{note["velocity"]}'
            
            events.append([
                event2word['tempo'].get('CONTI', 0),
                event2word['chord'].get('CONTI', 0),
                event2word['bar-beat'].get('CONTI', 0),
                event2word['type']['Note'],
                event2word['pitch'].get(pitch_str, 0),
                event2word['duration'].get(duration_str, 0),
                event2word['velocity'].get(velocity_str, 0)
            ])
    
    return np.array(events)

################################################################################
# Controlled Generation
################################################################################

def inference_with_control(model, dictionary, target_bars=8, prompt_midi=None, temperature=1.2, top_p=0.9, top_k=5):
    """
    Generate music with controllable number of bars and optional MIDI prompt
    
    Args:
        model: TransformerModel instance
        dictionary: (event2word, word2event) tuple
        target_bars: number of bars to generate
        prompt_midi: path to MIDI file for continuation (optional)
        temperature: sampling temperature
        top_p: nucleus sampling parameter
    
    Returns:
        numpy array of compound words
    """
    event2word, word2event = dictionary
    classes = word2event.keys()

    def print_word_cp(cp):
        result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
        for r in result:
            print('{:15s}'.format(str(r)), end=' | ')
        print('')

    with torch.no_grad():
        final_res = []
        memory = None
        h = None
        cnt_bar = 0
        
        # Initialize with prompt if provided
        if prompt_midi is not None:
            print('[*] Loading prompt from:', prompt_midi)
            prompt_events = read_midi_to_events(prompt_midi, word2event, event2word)
            
            # Count bars in prompt
            for event in prompt_events:
                if word2event['bar-beat'][event[2]] == 'Bar':
                    cnt_bar += 1
            
            print(f'[*] Prompt has {cnt_bar} bars, {len(prompt_events)} events')
            
            # Feed prompt through model
            prompt_tensor = torch.from_numpy(prompt_events).long().cuda()
            print('------ processing prompt ------')
            
            for step in range(len(prompt_events)):
                if step % 100 == 0:
                    print(f'Processing prompt event {step}/{len(prompt_events)}')
                
                input_ = prompt_tensor[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(prompt_events[step, :][None, ...])
                
                h, y_type, memory = model.forward_hidden(
                    input_, memory, is_training=False)
            
            print(f'[*] Prompt processed, starting generation from bar {cnt_bar}')
        
        else:
            # Initialize from scratch
            init = np.array([
                [0, 0, 1, 1, 0, 0, 0],  # bar
            ])
            
            cnt_bar = 1
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')
            
            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                
                h, y_type, memory = model.forward_hidden(
                    input_, memory, is_training=False)
        
        # Generate until target bars
        print(f'------ generating {target_bars} bars ------')
        max_tokens = target_bars * 200  # Safety limit: ~200 tokens per bar
        token_count = 0
        
        while cnt_bar < target_bars and token_count < max_tokens:
            # Sample next token
            next_arr = model.froward_output_sampling(h, y_type)
            final_res.append(next_arr[None, ...])
            
            print(f'bar: {cnt_bar}/{target_bars}', end='  ==')
            print_word_cp(next_arr)
            
            # Forward
            input_ = torch.from_numpy(next_arr).long().cuda()
            input_ = input_.unsqueeze(0).unsqueeze(0)
            h, y_type, memory = model.forward_hidden(
                input_, memory, is_training=False)
            
            # Update bar count
            if word2event['bar-beat'][next_arr[2]] == 'Bar':
                cnt_bar += 1
            
            token_count += 1
        
        # Add EOS token
        print('\n[*] Target bars reached, adding EOS')
        
    print('\n--------[Done]--------')
    final_res = np.concatenate(final_res)
    print(f'Generated {final_res.shape[0]} tokens for {cnt_bar} bars')
    return final_res

################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(description='Controlled Music Generation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to model checkpoint directory')
    parser.add_argument('--loss', type=int, required=True,
                        help='loss value of checkpoint to load')
    parser.add_argument('--dict_path', type=str, required=True,
                        help='path to dictionary.pkl')
    parser.add_argument('--output_dir', type=str, default='controlled_gen',
                        help='output directory for generated MIDIs')
    parser.add_argument('--num_bars', type=int, default=8,
                        help='number of bars to generate')
    parser.add_argument('--prompt_midi_dir', type=str, default=None,
                        help='path to MIDI file for continuation (optional)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='nucleus sampling parameter')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top-k sampling parameter')
    parser.add_argument('--gid', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gid)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dictionary
    print('[*] Loading dictionary from:', args.dict_path)
    dictionary = pickle.load(open(args.dict_path, 'rb'))
    event2word, word2event = dictionary
    
    # Get number of classes
    n_class = []
    for key in event2word.keys():
        n_class.append(len(event2word[key]))
    print('[*] Number of classes:', n_class)
    
    # Load model
    name = f'loss_{args.loss}'
    path_ckpt = os.path.join(args.model_path, name + '_params.pt')
    print('[*] Loading model from:', path_ckpt)
    
    model = TransformerModel(n_class, is_training=False)
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(path_ckpt))
    
    # Generate samples
    print(f'\n[*] Generating {args.num_samples} samples with {args.num_bars} bars each')
    if args.prompt_midi_dir:
        print(f'[*] Using prompt: {args.prompt_midi_dir}')
    
    if args.prompt_midi_dir is None:
        for i in range(args.num_samples):
            print(f'\n{"="*60}')
            print(f'Generating sample {i+1}/{args.num_samples}')
            print(f'{"="*60}')
            
            try:
                # Generate
                result = inference_with_control(
                    model=model,
                    dictionary=dictionary,
                    target_bars=args.num_bars,
                    prompt_midi=None,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
            
                # Save
                output_name = f'generated_{args.num_bars}bars_{i}.mid'
                output_path = os.path.join(args.output_dir, output_name)
                write_midi(result, output_path, word2event)
                print(f'[*] Saved to: {output_path}')
                
            except Exception as e:
                print(f'[!] Error generating sample {i}: {e}')
                continue
    else:
        for midi in os.listdir(args.prompt_midi_dir):
            if midi.endswith('.mid'):
                prompt_name = os.path.splitext(os.path.basename(midi))[0]
            output_name = f'continuation_{prompt_name}_{args.num_bars}bars.mid'
            output_path = os.path.join(args.output_dir, output_name)
            try:
                result = inference_with_control(
                    model=model,
                    dictionary=dictionary,
                    target_bars=args.num_bars,
                    prompt_midi=os.path.join(args.prompt_midi_dir, midi),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
                write_midi(result, output_path, word2event)
                print(f'[*] Saved to: {output_path}')
            except Exception as e:
                print(f'[!] Error generating continuation for {midi}: {e}')
                continue
        
    print(f'\n[*] Generation complete! Saved to {args.output_dir}')

if __name__ == '__main__':
    main()