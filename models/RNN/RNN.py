import torch
import torch.nn as nn
from torch.nn import functional as F
from .RecurrentLayer import RecurrentLayer


# generative RNN
class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.recurrent = RecurrentLayer(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        B, T, C = x.shape

        # we feed the model one token at a time right bc it can't process everything at once?
        # RNN process things sequentially?
        logits = []
        
        # we don't want to carry over information from completely diff sentences.
        # so reset the hidden states first
        self.recurrent.reset_hidden_states()

        for t in range(T):
            timestep = x[:, t, :] # B, C
            x_timestep = self.recurrent(timestep) # B, C
            timestep_logits = self.lm_head(x_timestep)

            logits.append(timestep_logits)
        
        logits = torch.stack(logits, dim=1) # B, T, C

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # default reduce is mean afaik
        else:
            loss=None

        return logits, loss

    def generate(self, x, max_new_tokens):
        pass