import torch.nn as nn
from .AttentionLayer import AttentionLayer
from .MLP import MLP
from .LayerNorm import LayerNorm

class EncoderBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.norm1 = LayerNorm(config.embedding_size, config.layer_norm_bias)
    self.attention = AttentionLayer(config) # can look at itself as it likes

    self.norm2 = LayerNorm(config.embedding_size, config.layer_norm_bias)
    self.mlp = MLP(config)

  def forward(self, x):
    x = self.norm1(x)
    x = x + self.attention(x)

    x = self.norm2(x)
    x = x + self.mlp(x)

    return x