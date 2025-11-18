import subprocess
import os
import time

model_to_inference = ['gpt2', 'transformer_xl']
epoch = [100, 150, 200]

for model in model_to_inference:
    for ep in epoch:
        start_time = time.time()
        print(f"Running inference for model: {model} at epoch: {ep}")
        cmd = [
            'python', 'main.py',
            '--device', 'cuda:3',
            '--ckp_folder', f'./ckps/{model}',
            '--inference',
            '--model_type', model,
            '--epoch_to_inference', str(ep),
            '--inference_output_path', f'./results/{model}_epoch_{ep}',
            '--num_inference_files', '20',
        ]
        subprocess.run(cmd)
        end_time = time.time()
        print(f"Inference for model: {model} at epoch: {ep} completed in {end_time - start_time:.2f} seconds.\n")