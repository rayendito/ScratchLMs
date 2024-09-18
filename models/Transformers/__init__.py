import torch.nn as nn

from models.shared_blocks.EncoderBlock import EncoderBlock
from models.shared_blocks.DecoderBlock import DecoderBlock
from models.shared_blocks.PositionalEncoding import PositionalEncoding

class Transformers(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.context_length is not None

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.positional_encoding = PositionalEncoding(config)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_blocks)])
        self.encoder_attention = nn.AttentionLayer(config)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(config, cross=True) for _ in range(config.n_blocks)])

        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        vocab_embd = self.embedding(idx)                # B, T, C
        pos_embedding = self.positional_encoding(T)     # T, C
        
        x = vocab_embd + pos_embedding                  # B, T, C
        
        n_enc_blocks = len(self.encoder_blocks)
        for i, enc_block in enumerate(self.encoder_blocks):
            if(i<n_enc_blocks-1):
                x = enc_block(x)
            else:
                last_enc_k, last_enc_v = enc_block.get_att_layer_k_v(x)
        
        

    
    def generate(self, idx, max_new_tokens):
        pass