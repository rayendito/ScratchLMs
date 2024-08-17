import torch
import torch.nn as nn
import math

# the original pytorch tutorial on transformers is deprecated??? lol
# but the positional encoding snippet is already all over soverfl so
# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch

class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len = 50000):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        # pre computed positional encoding, maybe bc it's a lot faster to do it this way
        # but ofc it's limited, but we can set it to really high
        
        position = torch.arange(max_len).unsqueeze(1)       # max_len, 1
    
        div_term = torch.exp(                               # embedding_size / 2
            torch.arange(0, config.embedding_size, 2) *     # equation says 2i, so its 0, 2, 4, 6, ...
            (-math.log(10000.0) / config.embedding_size)
        )
        # basically this will translate to 1/(10000)^(2i/d_model) ––– from the equation of pos enc.

        # each position will have it's own vector that's as big as the embd size
        pe = torch.zeros(max_len, config.embedding_size)    # max_len, embd_size
        
        # filling even embedding indices with sin-ed position*div_term
        pe[:, 0::2] = torch.sin(position * div_term)        # max_len, 1 * embedding_size/2
                                                            # max_len * embedding_size/2
        
        # filling odd embedding indices with cos-ed position*div_term
        pe[:, 1::2] = torch.cos(position * div_term)        # max_len, 1 * embedding_size/2
                                                            # max_len * embedding_size/2
        
        # static, so put as a buffer
        self.register_buffer('pe', pe)
    
    def forward(self, context_length):
        return self.dropout(self.pe[:context_length]) # context_length, 





