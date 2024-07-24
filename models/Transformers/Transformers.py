import torch.nn as nn

from models.general_layers.EncoderBlock import EncoderBlock
from models.general_layers.DecoderBlock import DecoderBlock

class Transformers(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.context_length is not None

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.positional = nn.Embedding(config.context_length, config.embedding_size)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_blocks)])
        self.encoder_attention = nn.AttentionLayer(config)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(config, cross=True) for _ in range(config.n_blocks)])

        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, idx):
        B, T = idx.shape
    
    def generate(self, idx, max_new_tokens):
        pass