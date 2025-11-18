# read npy log and convert to txt log
import numpy as np
import os

def convert_log(npy_log_path, txt_log_path):
    log_data = np.load(npy_log_path, allow_pickle=True)
    print(f"Loaded log data keys: {list(log_data)}")
    with open(txt_log_path, 'w') as f:
        for i, loss in enumerate(log_data):
            f.write(f"Epoch {i}: Loss = {loss}\n")

if __name__ == "__main__":
    npy_log_path = "ckps/transformer_xl/training_loss.npy"
    txt_log_path = "ckps/transformer_xl/training_loss.txt"
    if os.path.exists(npy_log_path):
        convert_log(npy_log_path, txt_log_path)
        print(f"Converted {npy_log_path} to {txt_log_path}")
    else:
        print(f"{npy_log_path} does not exist.")