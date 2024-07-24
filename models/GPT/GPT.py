import torch
import torch.nn as nn
from torch.nn import functional as F

from models.general_layers.DecoderBlock import DecoderBlock

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()

    # why check specifically these two?
    assert config.vocab_size is not None
    assert config.context_length is not None

    self.config = config
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_size)
    self.position_embedding_table = nn.Embedding(config.context_length, config.embedding_size)
    self.attn_blocks =  nn.ModuleList([DecoderBlock(config) for _ in range(config.n_blocks)])
    self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    vocab_embd_output = self.token_embedding_table(idx) # B T C
    positional_embd = self.position_embedding_table(torch.arange(T)) # T C
    x = vocab_embd_output + positional_embd # B T C
    
    for block in self.attn_blocks:
      x = block(x)
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
      logits, loss = self(idx[:, -self.config.context_length:])
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx