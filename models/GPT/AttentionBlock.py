import torch.nn as nn
from .CausalSelfAttention import CausalSelfAttention
from models.general_layers.MLP import MLP
from models.general_layers.LayerNorm import LayerNorm

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.norm1 = LayerNorm(config.embedding_size, config.layer_norm_bias)
    self.attention = CausalSelfAttention(config)
    self.norm2 = LayerNorm(config.embedding_size, config.layer_norm_bias)
    self.mlp = MLP(config)

  def forward(self, x):
    x = self.norm1(x)
    x = x + self.attention(x)

    x = self.norm2(x)
    x = x + self.mlp(x)

    return x