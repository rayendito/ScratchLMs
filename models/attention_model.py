import torch
import math
import torch.nn as nn
from torch.nn import functional as F

EMBEDDING_SIZE = 32
N_HEADS = 8 # ATT_HEAD_SIZE * N_HEADS has to be = EMBEDDING_SIZE

class CausalSelfAttention(nn.Module):
  def __init__(self, block_size):
    super().__init__()
    # self.query_layer = nn.Linear(EMBEDDING_SIZE, ATT_HEAD_SIZE, bias=False)
    # self.key_layer = nn.Linear(EMBEDDING_SIZE, ATT_HEAD_SIZE, bias=False)
    # self.value_layer = nn.Linear(EMBEDDING_SIZE, ATT_HEAD_SIZE, bias=False)
    assert EMBEDDING_SIZE % N_HEADS == 0
    self.c_attn = nn.Linear(EMBEDDING_SIZE, 3 * EMBEDDING_SIZE, bias= False)
    self.register_buffer('attention_mask', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape # B T C

    keys, queries, values = self.c_attn(x).split(EMBEDDING_SIZE, dim=-1)

    # keys = self.key_layer(x) # B T ATT_HEAD_SIZE
    # queries = self.query_layer(x) # B T ATT_HEAD_SIZE
    # values = self.value_layer(x) # B T ATT_HEAD_SIZE

    keys = keys.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # B, nh, T, head_size
    queries = queries.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # B, nh, T, head_size
    values = values.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # B, nh, T, head_size
    
    # attention_mask = torch.tril(torch.ones(T,T))
    
    # B, nh, T, head_size @ B, nh, head_size, T = B, nh, T, T
    att_weights = (queries @ keys.transpose(-2,-1)) * (1.0 / math.sqrt(keys.size(-1))) # B, nh, T, T
    att_weights = att_weights.masked_fill(self.attention_mask == 0, -float('inf'))
    att_weights = F.softmax(att_weights, dim=-1)
    y = att_weights @ values # B nh T T @ B, nh, T, head_size = B, nh, T, head_size
    y = y.transpose(1,2).contiguous().view(B, T, C) # basically just transpose 1 2 and concatenate all attention heads

    return y

class nanoGPT(nn.Module):
  def __init__(self, vocab_size, block_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_SIZE)
    self.position_embedding_table = nn.Embedding(block_size, EMBEDDING_SIZE)
    self.lm_head = nn.Linear(EMBEDDING_SIZE, vocab_size)
    self.self_attention = CausalSelfAttention(block_size)
    self.block_size = block_size

  def forward(self, idx, targets=None):
    B, T = idx.shape

    vocab_embd_output = self.token_embedding_table(idx) # B T C
    positional_embd = self.position_embedding_table(torch.arange(T)) # T C
    x = vocab_embd_output + positional_embd # B T C
    x = self.self_attention(x) # B T C 
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