import torch.nn as nn
from .RecurrentLayer import RecurrentLayer
from models.shared_blocks.LayerNorm import LayerNorm
from models.shared_blocks.MLP import MLP

# borrowing the idea from transformers
# after the recurrent layer, normalize, expand it to twice the size, activation function, project it back, norm again
class RecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.embedding_size, bias=False)
        self.recurrent = RecurrentLayer(config)
        self.norm2 = LayerNorm(config.embedding_size, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.recurrent(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x