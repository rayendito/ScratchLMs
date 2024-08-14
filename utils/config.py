from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size : int
    device : str
    context_length : int = 16
    embedding_size : int = 128
    n_attn_heads : int = 8
    n_blocks : int = 6
    layer_norm_bias : bool = False
    dropout : float = 0
    
@dataclass
class RNNConfig:
    vocab_size : int
    device : str
    context_length : int = 16
    embedding_size : int = 128
    n_blocks : int = 6
    layer_norm_bias : bool = False
    dropout : float = 0
    