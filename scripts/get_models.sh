#!/bin/bash

gdown 1kOcA6WshJJboZznO1AwSVvemy0q5zcnp -O models.zip
unzip models.zip
gdown 1W76tPTK381Fy2zJuAEirFft-sA8pocEO -O basic_event_dictionary.pkl

# gpt2/transformer-xl
mv ./models/gpt2/epoch_100.pkl ./ckps/gpt2/checkpoints
mv ./models/gpt2/epoch_150.pkl ./ckps/gpt2/checkpoints
mv ./models/gpt2/epoch_200.pkl ./ckps/gpt2/checkpoints
mv ./models/transformer_xl/epoch_100.pkl ./ckps/transformer_xl/checkpoints
mv ./models/transformer_xl/epoch_150.pkl ./ckps/transformer_xl/checkpoints
mv ./models/transformer_xl/epoch_200.pkl ./ckps/transformer_xl/checkpoints

# cpword
mv models/cpword/loss_30_params.pt compound-word-transformer/exp
mv models/cpword/loss_36_params.pt compound-word-transformer/exp
mv models/cpword/loss_50_params.pt compound-word-transformer/exp

# compose_embellish
mkdir -p Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400
mkdir -p Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2 
mv models/compose_embellish/ep040_loss0.809_optim.pt Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/optim/ep040_loss0.809_optim.pt
mv models/compose_embellish/ep200_loss0.576_optim.pt Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/optim/ep200_loss0.576_optim.pt
mv models/compose_embellish/ep040_loss0.809_params.pt Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/params/ep040_loss0.809_params.pt
mv models/compose_embellish/ep200_loss0.576_params.pt Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/params/ep200_loss0.576_params.pt

# rm -rf models.zip models