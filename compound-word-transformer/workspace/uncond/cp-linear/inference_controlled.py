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
import time

################################################################################
# Constants
################################################################################
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

################################################################################
# MIDI to Words Conversion
################################################################################

def read_midi_to_words(midi_path, event2word, word2event):
    """
    Read MIDI file and convert to compound word representation 
    Following the exact preprocessing pipeline: midi2corpus → corpus2events → events2words
    
    Args:
        midi_path: path to MIDI file
        event2word: mapping from event type -> event string -> integer
        word2event: mapping from event type -> integer -> event string
    
    Returns:
        numpy array of compound words (shape: [seq_len, 7])
        Each row: [tempo, chord, bar-beat, type, pitch, duration, velocity]
    """
    DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 64+1, dtype=np.int32)
    DEFAULT_BPM_BINS = np.linspace(32, 224, 64+1, dtype=np.int32)
    DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60+1, dtype=np.int32)
    MIN_VELOCITY = 40
    NOTE_SORTING = 1  # descending by pitch
    
    midi_obj = miditoolkit.MidiFile(midi_path)
    
    # === Step 1: Extract and quantize notes (matching midi2corpus.py) ===
    notes = []
    for instr in midi_obj.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                notes.append(note)
    
    if not notes:
        print('[Warning] No notes found')
        return np.array([[0, 0, 1, 1, 0, 0, 0]])
    
    # Sort notes: by start, then by pitch (descending)
    if NOTE_SORTING == 1:
        notes.sort(key=lambda x: (x.start, -x.pitch))
    else:
        notes.sort(key=lambda x: (x.start, x.pitch))
    
    # Compute offset (remove empty bars at beginning)
    first_note_time = notes[0].start
    last_note_time = notes[-1].start
    quant_time_first = int(np.round(first_note_time / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    
    # Determine actual bar range (from first bar with content to last)
    first_content_bar = quant_time_first // BAR_RESOL - offset
    last_content_bar = last_bar
    
    # Extract tempo/chord info
    tempos = [(t.time, int(t.tempo)) for t in midi_obj.tempo_changes]
    chords = [(m.time, m.text) for m in midi_obj.markers]
    if not tempos:
        tempos = [(0, 120)]
    
    # Quantize tempos
    tempos = [(t, DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - bpm))]) for t, bpm in tempos]
    
    # Process notes with quantization
    note_grid = {}
    for note in notes:
        # Adjust for offset
        start_adjusted = note.start - offset * BAR_RESOL
        end_adjusted = note.end - offset * BAR_RESOL
        
        if start_adjusted < 0:
            continue
        
        # Quantize start time
        quant_time = int(np.round(start_adjusted / TICK_RESOL) * TICK_RESOL)
        
        # Quantize velocity
        velocity_quant = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS - note.velocity))]
        velocity_quant = max(MIN_VELOCITY, velocity_quant)
        
        # Compute duration
        note_duration = end_adjusted - start_adjusted
        if note_duration > BAR_RESOL:
            note_duration = BAR_RESOL
        ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
        duration_quant = max(TICK_RESOL, ntick_duration)  # At least one tick
        
        # Store note info (don't modify original note object)
        note_info = {
            'pitch': note.pitch,
            'duration': duration_quant,
            'velocity': velocity_quant
        }
        
        # Group by quantized time
        if quant_time not in note_grid:
            note_grid[quant_time] = []
        note_grid[quant_time].append(note_info)
    
    # Process tempo grid
    tempo_grid = {}
    for t, tempo in tempos:
        t = t - offset * BAR_RESOL
        if t < 0:
            t = 0
        quant_time = int(np.round(t / TICK_RESOL) * TICK_RESOL)
        tempo_grid[quant_time] = tempo
    
    # Process chord grid  
    chord_grid = {}
    for t, chord in chords:
        t = t - offset * BAR_RESOL
        if t < 0:
            t = 0
        quant_time = int(np.round(t / TICK_RESOL) * TICK_RESOL)
        chord_grid[quant_time] = chord
    
    # === Step 2: Build events (matching corpus2events.py) ===
    events = []
    
    # Only iterate through bars that actually have content
    # Start from first_content_bar to avoid empty bars at beginning
    for bar_step in range(first_content_bar * BAR_RESOL, last_content_bar * BAR_RESOL, BAR_RESOL):
        # Bar event at start of each bar
        events.append({
            'tempo': 0,
            'chord': 0,
            'bar-beat': 'Bar',
            'type': 'Metrical',
            'pitch': 0,
            'duration': 0,
            'velocity': 0
        })
        
        # Process each beat position in this bar
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_text = f'Beat_{(timing - bar_step) // TICK_RESOL}'
            pos_events = []
            pos_on = False
            
            # Check what's at this timing
            t_tempos = tempo_grid.get(timing, None)
            t_chords = chord_grid.get(timing, None)
            t_notes = note_grid.get(timing, [])
            
            # Metrical event: only if tempo or chord change
            if t_tempos is not None or t_chords is not None:
                tempo_val = f'Tempo_{t_tempos}' if t_tempos is not None else 'CONTI'
                chord_val = t_chords if t_chords is not None else 'CONTI'
                
                pos_events.append({
                    'tempo': tempo_val,
                    'chord': chord_val,
                    'bar-beat': pos_text,
                    'type': 'Metrical',
                    'pitch': 0,
                    'duration': 0,
                    'velocity': 0
                })
                pos_on = True
            
            # Note events
            if len(t_notes) > 0:
                # If no metrical event yet, create one with CONTI
                if not pos_on:
                    pos_events.append({
                        'tempo': 'CONTI',
                        'chord': 'CONTI',
                        'bar-beat': pos_text,
                        'type': 'Metrical',
                        'pitch': 0,
                        'duration': 0,
                        'velocity': 0
                    })
                
                # Add all notes at this position
                for note_info in t_notes:
                    pos_events.append({
                        'tempo': 0,
                        'chord': 0,
                        'bar-beat': 0,
                        'type': 'Note',
                        'pitch': f'Note_Pitch_{note_info["pitch"]}',
                        'duration': f'Note_Duration_{note_info["duration"]}',
                        'velocity': f'Note_Velocity_{note_info["velocity"]}'
                    })
            
            # Only add if there are events at this position
            if len(pos_events) > 0:
                events.extend(pos_events)
    
    # BAR ending (one more bar event at the very end)
    events.append({
        'tempo': 0,
        'chord': 0,
        'bar-beat': 'Bar',
        'type': 'Metrical',
        'pitch': 0,
        'duration': 0,
        'velocity': 0
    })
    
    # EOS
    events.append({
        'tempo': 0,
        'chord': 0,
        'bar-beat': 0,
        'type': 'EOS',
        'pitch': 0,
        'duration': 0,
        'velocity': 0
    })
    
    # === Step 3: Convert events to words (matching events2words.py) ===
    class_keys = ['tempo', 'chord', 'bar-beat', 'type', 'pitch', 'duration', 'velocity']
    words = []
    
    for event in events:
        word = [event2word[key].get(event[key], 0) for key in class_keys]
        words.append(word)
    
    result = np.array(words, dtype=np.int64)
    print(f'[Info] Converted MIDI to {len(result)} compound words, {last_bar} bars, {len(note_grid)} positions')
    return result


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
        generated_only = []  # Track only generated events (without prompt)
        memory = None
        h = None
        cnt_bar = 0
        prompt_length = 0  # Track where prompt ends
        
        # Initialize with prompt if provided
        if prompt_midi is not None:
            print('[*] Loading prompt from:', prompt_midi)
            prompt_events = read_midi_to_words(prompt_midi, event2word, word2event)
            # Count bars in prompt
            for event in prompt_events:
                if word2event['bar-beat'][event[2]] == 'Bar':
                    cnt_bar += 1
            
            print(f'[*] Prompt has {cnt_bar} bars, {len(prompt_events)} events')
            prompt_length = len(prompt_events)
            
            # Feed prompt through model AND add to final output
            prompt_tensor = torch.from_numpy(prompt_events).long().cuda()
            print('------ processing prompt ------')
            
            for step in range(len(prompt_events)):
                if step % 100 == 0:
                    print(f'Processing prompt event {step}/{len(prompt_events)}')
                
                input_ = prompt_tensor[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(prompt_events[step, :][None, ...])  # Keep prompt in final output
                
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
        print(f'------ generating to reach {target_bars} total bars ------')
        max_tokens = target_bars * 200  # Safety limit: ~200 tokens per bar
        token_count = 0
        
        while cnt_bar < target_bars and token_count < max_tokens:
            # Sample next token
            next_arr = model.froward_output_sampling(h, y_type)
            final_res.append(next_arr[None, ...])
            generated_only.append(next_arr[None, ...])  # Track generated portion
            # print(final_res)
            
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
    if prompt_midi is not None:
        print(f'Total: {final_res.shape[0]} tokens ({prompt_length} prompt + {len(generated_only)} generated) for {cnt_bar} bars')
    else:
        print(f'Generated {final_res.shape[0]} tokens for {cnt_bar} bars')
    return final_res[1:]    # Remove the first extra bar (for some ugly reason)

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
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds")