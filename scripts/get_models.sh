#!/bin/bash

gdown 1eaqKD9ZMDAMV3F54uKi_XiY80cJ6cEH3 -O models.zip
unzip models.zip -d ./models
gdown 1W76tPTK381Fy2zJuAEirFft-sA8pocEO -O basic_event_dictionary.pkl

# gpt2/transformer-xl
mv ./models/gpt2/epoch_100.pkl ./ckps/gpt2/checkpoints
mv ./models/gpt2/epoch_150.pkl ./ckps/gpt2/checkpoints
mv ./models/gpt2/epoch_200.pkl ./ckps/gpt2/checkpoints
mv ./models/transformer-xl/epoch_100.pkl ./ckps/transformer-xl/checkpoints
mv ./models/transformer-xl/epoch_150.pkl ./ckps/transformer-xl/checkpoints
mv ./models/transformer-xl/epoch_200.pkl ./ckps/transformer-xl/checkpoints

# cpword
mv models/cpword/loss_30_params.pt compound-word-transformer/exp
mv models/cpword/loss_36_params.pt compound-word-transformer/exp
mv models/cpword/loss_50_params.pt compound-word-transformer/exp

# compose_embellish
mv models/compose_embellish/ep040_loss0.809_optim.pt Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/optim
mv models/compose_embellish/ep200_loss0.576_optim.pt Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/optim

rm -rf models.zip models