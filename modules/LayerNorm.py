import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
  def __init__(self, ndim, bias):
    super().__init__()
    self.weights = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

  def forward(self, input):
    return F.layer_norm(input, self.weights.shape, self.weights, self.bias, 1e-5)