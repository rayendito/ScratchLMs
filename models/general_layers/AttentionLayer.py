import torch
import math
import torch.nn as nn
from torch.nn import functional as F

# DEFAULT: Causal self attention (masked with tril, attend to self)
# set causal=False to disable masking, and add keys and values (e.g. from an encoder) to make a cross attention
class AttentionLayer(nn.Module):
  def __init__(self, config, causal = False, cross=False):
    super().__init__()
    assert config.embedding_size % config.n_attn_heads == 0
    self.config = config
    self.c_attn = nn.Linear(config.embedding_size, 3 * config.embedding_size, bias= False) # 3 because Q K V
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)
    self.proj_layer = nn.Linear(config.embedding_size, config.embedding_size)
    self.causal = causal
    self.cross = cross

  def forward(self, x, cross_attn_key = None, cross_attn_value = None):
    B, T, C = x.shape # B T C
    queries, keys, values = self.c_attn(x).split(self.config.embedding_size, dim=-1)
    queries = queries.view(B, T, self.config.n_attn_heads, C // self.config.n_attn_heads).transpose(1, 2) # B, nh, T, head_size
    
    if(self.cross):
      assert cross_attn_key and cross_attn_value
      keys = cross_attn_key
      values = cross_attn_value
    else:
      # if self.cross_attn_key and self.cross_attn_value are None, assume self-attention
      keys = keys.view(B, T, self.config.n_attn_heads, C // self.config.n_attn_heads).transpose(1, 2) # B, nh, T, head_size
      values = values.view(B, T, self.config.n_attn_heads, C // self.config.n_attn_heads).transpose(1, 2) # B, nh, T, head_size
    
    # B, nh, T, head_size @ B, nh, head_size, T = B, nh, T, T
    att_weights = (queries @ keys.transpose(-2,-1)) * (1.0 / math.sqrt(keys.size(-1))) # B, nh, T, T
    
    if self.causal: # on decoders, mask to prevent attending to future tokens
      attention_mask = torch.tril(torch.ones(T, T, requires_grad=False))
      att_weights = att_weights.masked_fill(attention_mask == 0, -float('inf')) # B, nh, T, T
    
    att_weights = F.softmax(att_weights, dim=-1)
    att_weights = self.attn_dropout(att_weights) # dropout in attention
    
    # wait but doesn't this att layer has a zero upper triangle?
    # what if we apply zeros or scale by 1/1-p only if it's not 0?
    
    y = att_weights @ values # (B nh T T) @ (B, nh, T, head_size) = (B, nh, T, head_size)
    y = y.transpose(1,2).contiguous().view(B, T, C) # basically just transpose 1 2 and concatenate all attention heads
    y = self.proj_layer(y)
    y = self.resid_dropout(y) # projection layer (basically 'aggregating' the attention heads) and dropout regularization

    return y
  
  def get_key_and_value(self, x):
    B, T, C = x.shape # B T C
    _, keys, values = self.c_attn(x).split(self.config.embedding_size, dim=-1)
    keys = keys.view(B, T, self.config.n_attn_heads, C // self.config.n_attn_heads).transpose(1, 2) # B, nh, T, head_size
    values = values.view(B, T, self.config.n_attn_heads, C // self.config.n_attn_heads).transpose(1, 2) # B, nh, T, head_size
    
    return keys, values