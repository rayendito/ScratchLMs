import torch
import math
import torch.nn as nn
from torch.nn import functional as F

# TODO: make a config class. passing around block_size is very bad design :')
# also be careful, block_size here is really context_length, not the same with attention blocks lol

EMBEDDING_SIZE = 32
N_HEADS = 8 # ATT_HEAD_SIZE * N_HEADS has to be = EMBEDDING_SIZE
NORM_BIAS = False
DROPOUT = 0.0

# TODO: i honestly don't get why we should normalize. maybe i should rewatch
# watched it, but he doesn't really explain why lmao
class LayerNorm(nn.Module):
  def __init__(self, ndim, bias):
    super().__init__()
    self.weights = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

  def forward(self, input):
    return F.layer_norm(input, self.weights.shape, self.weights, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    assert EMBEDDING_SIZE % N_HEADS == 0

    self.c_attn = nn.Linear(EMBEDDING_SIZE, 3 * EMBEDDING_SIZE, bias= False)
    self.attn_dropout = nn.Dropout(DROPOUT)
    self.resid_dropout = nn.Dropout(DROPOUT)
    self.proj_layer = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
    self.register_buffer('attention_mask', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape # B T C

    keys, queries, values = self.c_attn(x).split(EMBEDDING_SIZE, dim=-1)

    keys = keys.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # B, nh, T, head_size
    queries = queries.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # B, nh, T, head_size
    values = values.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # B, nh, T, head_size
    
    # B, nh, T, head_size @ B, nh, head_size, T = B, nh, T, T
    att_weights = (queries @ keys.transpose(-2,-1)) * (1.0 / math.sqrt(keys.size(-1))) # B, nh, T, T
    att_weights = att_weights.masked_fill(self.attention_mask == 0, -float('inf'))
    att_weights = F.softmax(att_weights, dim=-1)
    att_weights = self.attn_dropout(att_weights) # dropout in attention
    # wait but doesn't this att layer has a zero upper triangle?
    # what if we apply zeros or scale by 1/1-p only if it's not 0?

    y = att_weights @ values # B nh T T @ B, nh, T, head_size = B, nh, T, head_size
    y = y.transpose(1,2).contiguous().view(B, T, C) # basically just transpose 1 2 and concatenate all attention heads
    y = self.proj_layer(y)
    y = self.resid_dropout(y) # projection layer (basically 'aggregating' the attention heads) and dropout regularization

    return y

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    # i'm guessing this is so that the model can have like some capacity to understand complexities in GELU?
    self.ffnn = nn.Linear(EMBEDDING_SIZE, 4*EMBEDDING_SIZE)
    self.gelu = nn.GELU()
    self.proj = nn.Linear(4*EMBEDDING_SIZE, EMBEDDING_SIZE)
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    x = self.ffnn(x)
    x = self.gelu(x)
    x = self.proj(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    self.norm1 = LayerNorm(EMBEDDING_SIZE, NORM_BIAS)
    self.attention = CausalSelfAttention(block_size)
    self.norm2 = LayerNorm(EMBEDDING_SIZE, NORM_BIAS)
    self.mlp = MLP()

  def forward(self, x):
    x = self.norm1(x)
    x = x + self.attention(x)

    x = self.norm2(x)
    x = x + self.mlp(x)

    return x

class nanoGPT(nn.Module):
  def __init__(self, vocab_size, block_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_SIZE)
    self.position_embedding_table = nn.Embedding(block_size, EMBEDDING_SIZE)
    self.block = Block(block_size)
    self.lm_head = nn.Linear(EMBEDDING_SIZE, vocab_size)
    # self.self_attention = CausalSelfAttention(block_size)
    self.block_size = block_size

  def forward(self, idx, targets=None):
    B, T = idx.shape

    vocab_embd_output = self.token_embedding_table(idx) # B T C
    positional_embd = self.position_embedding_table(torch.arange(T)) # T C
    x = vocab_embd_output + positional_embd # B T C
    x = self.block(x) # B T C
    logits = self.lm_head(x)
  
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    # i think it sorta makes sense that it returns logits (the result)
    # and the loss (a measure of how good the results are)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is the current sentence
    # max_new_tokens is how much you want it to yap
    for _ in range(max_new_tokens):
      logits, loss = self(idx[:, -self.block_size:])
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx