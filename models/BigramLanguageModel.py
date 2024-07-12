import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size, embedding_size, block_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
    self.position_embedding_table = nn.Embedding(block_size, embedding_size)
    self.lm_head = nn.Linear(embedding_size, vocab_size)
    self.block_size = block_size

  def forward(self, idx, targets=None):
    B, T = idx.shape

    vocab_embd_output = self.token_embedding_table(idx) # B T C
    positional_embd = self.position_embedding_table(torch.arange(T)) # T C
    x = vocab_embd_output + positional_embd
    logits = self.lm_head(x)
    pass
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