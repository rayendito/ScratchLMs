import torch
import torch.nn as nn

# simple recurrent network (Elman) –– https://en.wikipedia.org/wiki/Recurrent_neural_network
class RecurrentLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # idk i cameup with this resizing myself lol
        self.hidden_size = config.embedding_size*2 # i think my model overfits very fast with this one lol
        self.hidden_state = None

        self.input_weights = nn.Linear(config.embedding_size, self.hidden_size)
        self.hidden_weights = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_weights = nn.Linear(self.hidden_size, config.embedding_size)

        self.activation = nn.Tanh()

    def forward(self, x):
        B, C = x.shape # C = embedding_size
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(B, self.hidden_size, requires_grad=False)
            
        wx = self.input_weights(x) # B, hidden_size
        ch = self.hidden_weights(self.hidden_state) # B, hidden_size
        self.hidden_state = self.activation(wx + ch) # B, hidden_size 
        # self.hidden_state.requires_grad_(False)
        
        y = self.activation(self.output_weights(self.hidden_state)) # B embedding size

        return y

    def reset_hidden_states(self):
        self.hidden_state = None

