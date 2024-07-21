from dataclasses import dataclass

@dataclass
class Config:
    vocab_size : int
    context_length : int
    embedding_size : int
    n_attn_heads : int
    n_blocks : int
    layer_norm_bias : bool
    dropout : float
