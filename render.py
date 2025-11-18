import os
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MIDI to WAV Renderer")
    parser.add_argument('--midi_dir', type=str, help='Directory containing MIDI files to be converted.')
    parser.add_argument('--store_dir', type=str, help='Directory to store the generated WAV files.')
    parser.add_argument('--soundfont_path', type=str, default='TimGM.sf2', help='Path to the SoundFont file.')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate for the output WAV file.')
    return parser.parse_args()

def midi_to_wav(midi_path, output_wav_path="generated.wav", soundfont_path="TimGM.sf2", sample_rate=44100):
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

def get_filelist(dir, ext=".mid"):
    filelist = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(ext):
                filelist.append(os.path.join(root, file))
    return filelist

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.store_dir, exist_ok=True)
    for path in get_filelist(args.midi_dir, ext=".mid"):
        midi_to_wav(
            midi_path=path,
            output_wav_path=os.path.join(
                args.store_dir,
                os.path.splitext(os.path.basename(path))[0] + ".wav"
            ),
            soundfont_path=args.soundfont_path,
            sample_rate=args.sample_rate
        )