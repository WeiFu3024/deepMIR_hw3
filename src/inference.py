# Inference script for MIDI generation
import os
import torch
import numpy as np
from miditok import REMI, CPWord, TokenizerConfig
from symusic import Score

from src.model import Model
from src.utils import *

def load_model(config_path, use_epoch=None):
    args = load_config(config_path)
    use_epoch = args.train.epochs if use_epoch is None else use_epoch
    tokenizer_params = {
        "pitch_range": args.data.tokenizer.pitch_range,
        "beat_res": args.data.tokenizer.beat_res,
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
    set_seed(args.env.seed)
    
    model = Model(model_type=args.model.type, vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(args.train.save_dir, f'model_epoch_{use_epoch}.pth'), map_location=args.env.device))
    model.to(args.env.device)
    model.eval()
    print(f"Model loaded from '{os.path.join(args.train.save_dir, f'model_epoch_{use_epoch}.pth')}'")
    return model, tokenizer, args


def get_bar_token_name(tokenizer, midi_rep):
    """Get the bar token name based on MIDI representation"""
    if midi_rep == 'REMI':
        return 'Bar_None'
    elif midi_rep == 'CPWord':
        return 'Bar_None'  # CPWord also uses Bar_None
    else:
        raise ValueError(f"Unknown MIDI representation: {midi_rep}")


def count_bars_in_tokens(tokens, tokenizer, midi_rep):
    """Count number of bars in a token sequence"""
    bar_token_name = get_bar_token_name(tokenizer, midi_rep)
    bar_token_id = tokenizer.vocab[bar_token_name]
    
    bar_count = 0
    for token in tokens:
        if token == bar_token_id:
            bar_count += 1
    return bar_count


def create_initial_prompt(tokenizer, midi_rep, use_chords=False):
    """
    Create initial prompt tokens to start generation
    
    For REMI: [Bar, Position, Tempo/Chord, ...]
    For CPWord: [Bar, Position, Pitch, Velocity, Duration] (compound token)
    """
    vocab = tokenizer.vocab
    tokens = []
    
    # Start with Bar token
    bar_token_name = get_bar_token_name(tokenizer, midi_rep)
    tokens.append(vocab[bar_token_name])
    
    if midi_rep == 'REMI':
        # REMI: sequential tokens
        tokens.append(vocab['Position_0'])
        
        if use_chords:
            # Add random chord
            chord_tokens = [v for k, v in vocab.items() if 'Chord' in k and 'None' not in k]
            if chord_tokens:
                tokens.append(np.random.choice(chord_tokens))
        
        # Add tempo
        tempo_tokens = [v for k, v in vocab.items() if 'Tempo' in k]
        if tempo_tokens:
            tokens.append(np.random.choice(tempo_tokens))
            
    elif midi_rep == 'CPWord':
        # CPWord: compound tokens (Pitch-Velocity-Duration in one token)
        # Add position first
        tokens.append(vocab['Position_0'])
        
        # For CPWord, we might need to add a compound token
        # This depends on the exact CPWord vocabulary structure
        # We'll add a simple note to start
        try:
            # Try to find a middle C (60) compound token
            cpword_tokens = [k for k in vocab.keys() if 'Pitch_60' in k and 'Velocity' in k and 'Duration' in k]
            if cpword_tokens:
                tokens.append(vocab[cpword_tokens[0]])
            else:
                # Fallback: just use position
                pass
        except:
            pass
    
    return tokens


def temperature_sampling(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample from logits with temperature, top-k, and top-p (nucleus) sampling
    
    Args:
        logits: numpy array of logits
        temperature: temperature for scaling (higher = more random)
        top_k: if > 0, only sample from top k tokens
        top_p: if < 1.0, only sample from tokens with cumulative probability <= top_p
    """
    # Sanitize logits
    logits = np.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
    
    # Apply temperature
    logits = logits / max(temperature, 1e-8)
    
    # Convert to probabilities
    probs = np.exp(logits - np.max(logits))  # Numerical stability
    probs = probs / probs.sum()
    
    # Apply top-k filtering
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        filtered_probs = np.zeros_like(probs)
        filtered_probs[top_k_indices] = probs[top_k_indices]
        probs = filtered_probs / filtered_probs.sum()
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Find cutoff index
        cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
        cutoff_idx = max(1, min(cutoff_idx, len(sorted_probs)))
        
        # Keep only nucleus tokens
        nucleus_indices = sorted_indices[:cutoff_idx]
        filtered_probs = np.zeros_like(probs)
        filtered_probs[nucleus_indices] = probs[nucleus_indices]
        probs = filtered_probs / filtered_probs.sum()
    
    # Sample
    token_id = np.random.choice(len(probs), p=probs)
    return int(token_id)


def generate(
    model,
    tokenizer,
    args,
    n_target_bars=16,
    max_seq_len=1024,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    prompt_tokens=None,
    output_path='generated.mid'
):
    """
    Generate MIDI with controllable number of bars
    
    Args:
        model: trained model
        tokenizer: miditok tokenizer (REMI or CPWord)
        args: configuration args
        n_target_bars: target number of bars to generate
        max_seq_len: maximum sequence length (model's positional embedding limit)
        temperature: sampling temperature
        top_k: top-k sampling parameter
        top_p: nucleus sampling parameter
        prompt_tokens: optional list of initial tokens
        output_path: path to save generated MIDI
    """
    device = args.env.device
    model.eval()
    
    # Get bar token information
    bar_token_name = get_bar_token_name(tokenizer, args.data.midi_rep)
    bar_token_id = tokenizer.vocab[bar_token_name]
    vocab_size = len(tokenizer)
    
    print(f"Generating {n_target_bars} bars using {args.data.midi_rep} tokenizer...")
    print(f"Bar token: {bar_token_name} (id: {bar_token_id})")
    
    # Initialize prompt
    if prompt_tokens is None:
        tokens = create_initial_prompt(
            tokenizer, 
            args.data.midi_rep,
            use_chords=args.data.tokenizer.use_chords
        )
    else:
        tokens = list(prompt_tokens)
    
    print(f"Starting with {len(tokens)} prompt tokens")
    
    current_bars = count_bars_in_tokens(tokens, tokenizer, args.data.midi_rep)
    print(f"Initial bars in prompt: {current_bars}")
    
    # Generation loop
    with torch.no_grad():
        step = 0
        max_steps = 10000  # Safety limit
        
        while current_bars < n_target_bars and step < max_steps:
            # Prepare input (take last max_seq_len tokens to fit in model)
            if len(tokens) > max_seq_len:
                input_tokens = tokens[-max_seq_len:]
            else:
                input_tokens = tokens
            
            # Convert to tensor
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            
            # Forward pass
            if args.model.type == 'gpt2':
                logits = model(input_tensor).logits
            elif args.model.type == 'transformer_xl':
                # Transformer-XL returns tuple (logits, mems)
                outputs = model(input_tensor)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits
            else:
                logits = model(input_tensor)
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :].cpu().numpy()
            
            # Sample next token
            next_token = temperature_sampling(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Bounds check
            if next_token >= vocab_size or next_token < 0:
                print(f"\nWarning: Sampled token {next_token} out of bounds, clamping...")
                next_token = max(0, min(next_token, vocab_size - 1))
            
            # Add to sequence
            tokens.append(next_token)
            
            # Check if we generated a bar token
            if next_token == bar_token_id:
                current_bars += 1
                print(f"\rGenerated bars: {current_bars}/{n_target_bars}", end="")
            
            step += 1
        
        print(f"\nGeneration completed: {current_bars} bars in {len(tokens)} tokens")
    
    # Convert tokens to MIDI
    try:
        # Create TokSequence for miditok
        from miditok import TokSequence
        
        # Remove any padding tokens if present
        tokens = [t for t in tokens if t < vocab_size]
        
        tok_seq = TokSequence(ids=tokens)
        
        # Convert to MIDI
        midi = tokenizer([tok_seq])
        
        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        midi.dump_midi(output_path)
        print(f"MIDI saved to: {output_path}")
        
        return tokens
        
    except Exception as e:
        print(f"Error converting tokens to MIDI: {e}")
        # Save tokens as backup
        backup_path = output_path.replace('.mid', '_tokens.npy')
        np.save(backup_path, np.array(tokens))
        print(f"Tokens saved to: {backup_path}")
        return tokens


def generate_with_prompt(
    model,
    tokenizer,
    args,
    prompt_midi_path,
    n_additional_bars=8,
    **kwargs
):
    """
    Generate continuation of a prompt MIDI file
    
    Args:
        model: trained model
        tokenizer: miditok tokenizer
        args: configuration args
        prompt_midi_path: path to prompt MIDI file
        n_additional_bars: number of bars to generate after prompt
        **kwargs: additional arguments for generate()
    """
    # Load and tokenize prompt
    prompt_score = Score(prompt_midi_path)
    prompt_tok_seq = tokenizer(prompt_score)
    
    # Extract tokens from all tracks
    prompt_tokens = []
    for seq in prompt_tok_seq:
        prompt_tokens.extend(seq.ids)
    
    # Count bars in prompt
    current_bars = count_bars_in_tokens(prompt_tokens, tokenizer, args.data.midi_rep)
    print(f"Prompt contains {current_bars} bars")
    
    # Generate continuation
    target_bars = current_bars + n_additional_bars
    
    return generate(
        model=model,
        tokenizer=tokenizer,
        args=args,
        n_target_bars=target_bars,
        prompt_tokens=prompt_tokens,
        **kwargs
    )


def run_inference(config_path, use_epoch=None, output_dir='./results', **gen_kwargs):
    """
    Main inference function
    
    Args:
        config_path: path to config file
        use_epoch: which epoch checkpoint to use
        output_dir: directory to save results
        **gen_kwargs: keyword arguments for generate() function
    """
    # Load model
    model, tokenizer, args = load_model(config_path, use_epoch=use_epoch)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate
    output_path = os.path.join(output_dir, f'generated_{args.data.midi_rep}.mid')
    
    tokens = generate(
        model=model,
        tokenizer=tokenizer,
        args=args,
        output_path=output_path,
        **gen_kwargs
    )
    
    return tokens


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MIDI using trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epoch', type=int, default=None, help='Which epoch to use')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--n_bars', type=int, default=16, help='Number of bars to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--prompt', type=str, default=None, help='Path to prompt MIDI file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Run inference
    if args.prompt:
        model, tokenizer, cfg = load_model(args.config, use_epoch=args.epoch)
        tokens = generate_with_prompt(
            model=model,
            tokenizer=tokenizer,
            args=cfg,
            prompt_midi_path=args.prompt,
            n_additional_bars=args.n_bars,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            output_path=os.path.join(args.output_dir, 'generated_with_prompt.mid')
        )
    else:
        tokens = run_inference(
            config_path=args.config,
            use_epoch=args.epoch,
            output_dir=args.output_dir,
            n_target_bars=args.n_bars,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    print("Inference completed!")