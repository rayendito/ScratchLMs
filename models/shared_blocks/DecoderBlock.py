import torch.nn as nn
from .AttentionLayer import AttentionLayer
from .MLP import MLP
from .LayerNorm import LayerNorm

# based on the famous Transformers diagram, there are 2 attention layers
# first one is causal attention layer
# second is the cross attention layer
# gonna make the cross attention layer default to False so that nanoGPT doesn't change
class DecoderBlock(nn.Module):
  def __init__(self, config, cross = False):
    super().__init__()
    self.cross = cross

    # 1. Masked Multi Head Attention (toggle causal)
    self.norm1 = LayerNorm(config.embedding_size, config.layer_norm_bias)
    self.self_attention = AttentionLayer(config, causal=True, cross=False)

    # 2. Cross Attention (GPT does not need this, therefore optional)
    if(self.cross):
      self.norm2 = LayerNorm(config.embedding_size, config.layer_norm_bias)
      self.cross_attention = AttentionLayer(config, causal=False, cross=True)
      # ^no mask because it can look at the input as much as it likes
    
    # 3. FFNN
    self.norm3 = LayerNorm(config.embedding_size, config.layer_norm_bias)
    self.mlp = MLP(config)

  def forward(self, x, cross_attn_key=None, cross_attn_value=None):    
    x = self.norm1(x)
    x = x + self.self_attention(x)
    
    if(self.cross):
      assert cross_attn_key and cross_attn_value
      x = self.norm2(x)
      x = x + self.cross_attention(x, cross_attn_key, cross_attn_value)
    
    x = self.norm3(x)
    x = x + self.mlp(x)

    return x