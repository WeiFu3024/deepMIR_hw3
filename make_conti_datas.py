import os

dir_list = ['Pop1K7/midi_analyzed/src_001', 'Pop1K7/midi_analyzed/src_002', 'Pop1K7/midi_analyzed/src_003', 'Pop1K7/midi_analyzed/src_004']

def make_continuous_dir(dir_list, target_dir):
    file_list = []
    for dir in dir_list:
        files = os.listdir(dir)
        files = [os.path.join(dir, f) for f in files if f.endswith('.mid')]
        file_list.extend(files)
    
    for idx, file in enumerate(file_list):
        target_path = os.path.join(target_dir, f'src_{idx:04d}.mid')
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.symlink(os.path.abspath(file), target_path)

if __name__ == "__main__":
    make_continuous_dir(dir_list, './Pop1K7/continuous_src')