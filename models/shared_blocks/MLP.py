import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    # i'm guessing this is so that the model can have like some capacity to understand complexities in GELU?
    self.ffnn = nn.Linear(config.embedding_size, 4*config.embedding_size)
    self.gelu = nn.GELU()
    self.proj = nn.Linear(4*config.embedding_size, config.embedding_size)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    x = self.ffnn(x)
    x = self.gelu(x)
    x = self.proj(x)
    x = self.dropout(x)
    return x