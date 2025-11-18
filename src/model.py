import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, TransfoXLLMHeadModel, TransfoXLConfig


class Model(nn.Module):
    def __init__(self, model_type='gpt2', vocab_size=512):
        super(Model, self).__init__()
        #################################################
        # Support GPT2 and Transformer-XL
        #################################################
        self.model_type = model_type
        if model_type == 'gpt2':
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=1024,
                n_embd=768,
                n_layer=12,
                n_head=12
            )
            self.model = GPT2LMHeadModel(config)
        elif model_type == 'transformer_xl':
            config = TransfoXLConfig(
                vocab_size=vocab_size,
                d_model=512,
                n_layer=12,
                n_head=8,
                cutoffs=[]
            )
            self.model = TransfoXLLMHeadModel(config)


if __name__ == "__main__":
    vocab_size = 1000   # Just an example vocab size, replace with actual size of tokenizer
    model = Model(model_type='gpt2', vocab_size=vocab_size)
    print(model)