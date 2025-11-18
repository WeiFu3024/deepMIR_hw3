# DeepMIR HW3
This is the repository of the Homework 3 of DeepMIR (by R14942090), which focus on midi-generation model trained on the Pop1K7 dataset.

## Installation
It is recommanded to run this repository under `python 3.10`. Also recommand you to use conda or mamba to manage your enviroment.
Run `pip install -r requirements.txt` for necessary dependencies. (for gpt2, transformers-XL, and MusDr evaluation)

If you use conda (or mamba), you should run `conda install -c conda-forge fluidsynth` (or mamba, or sudo apt-get) for fluidsynth to render the output midi files to wavs.

If you want to inference the compound-word-transformer and Compose&Embellish model, you should also create enviroment based on the requirements in the original repo of compound-word-transformer and Compose&Embellish.

## Setup
If you want to install the Pop1K7 dataset, run `srcipts/get_dataset.sh` to get the dataset. If you want to get model checkpoints, run `scripts/get_models.sh`. So that all the trained models will be downloaded.

## Inference
There are several scripts in the `scripts` folder, for inferencing GPT2/Transformers based model, you can refer to `bash scripts/inference_remi.sh`. For inferencing Compound Word Transformer, you should refer to `bash scripts/inference_cpword.sh`. For inferencing Compose&Embellish, you should refer to `bash scripts/inference_compose_embellish.sh`. As for evaluation, run `bash srcipts/evaluation.sh`. Note that directly run the bash script might not work, since the bash scripts are designed to run under specific environment. You should deal with the environment based on the instructions in the original repos (including the installation session above for this repo) of compose_and_embellish and compound word transformer to create correct environment.  

All inference results and evaluation results will be stored into the `results` folder.