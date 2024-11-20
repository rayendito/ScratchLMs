import torch.nn as nn


# 3.1. Parameterizing Senses
class SenseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sense_layer = nn.Linear(config.embedding_size, config.n_senses * config.embedding_size)
    
    def forward(self, x):
        B, T, C = x.shape
        


class BackpackLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_size)

    def forward(self, x):
        B, T = x.shape        
        
        x = self.embedding_layer(x) # B, T, embd_dim
        




