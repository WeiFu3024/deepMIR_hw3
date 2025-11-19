#!/bin/bash

gdown 1NirbbSHtJZY63w1CutmRErUiv8qQ5quS -O models.zip
unzip models.zip
gdown 1W76tPTK381Fy2zJuAEirFft-sA8pocEO -O basic_event_dictionary.pkl
gdown 1ePR8rqp5vuoqTMdN0IrmQF_NUY-ewuwL -O remi_jtrans_vocab.pkl
gdown 1nQrZMmsg4sINdDXRCQ0sowFzHnv_EW2i -O skyline2midi_vocab.pkl

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
mkdir -p Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/optim
mkdir -p Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/params
mkdir -p Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/optim
mkdir -p Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/params
mkdir -p Compose_and_Embellish/stage01_compose/vocab
mkdir -p Compose_and_Embellish/stage02_embellish/vocab

mv models/compose_embellish/ep040_loss0.809_optim.pt Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/optim/ep040_loss0.809_optim.pt
mv models/compose_embellish/ep200_loss0.576_optim.pt Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/optim/ep200_loss0.576_optim.pt
mv models/compose_embellish/ep040_loss0.809_params.pt Compose_and_Embellish/ckpt/stage01_compose_finetune_pop1k7_seq2400/params/ep040_loss0.809_params.pt
mv models/compose_embellish/ep200_loss0.576_params.pt Compose_and_Embellish/ckpt/stage02_embellish_pop1k7_seq2048_gpt2/params/ep200_loss0.576_params.pt

mv remi_jtrans_vocab.pkl Compose_and_Embellish/stage01_compose/vocab/remi_jtrans_vocab.pkl
mv skyline2midi_vocab.pkl Compose_and_Embellish/stage02_embellish/vocab/skyline2midi_vocab.pkl

# rm -rf models.zip models