import torch
import torch.nn as nn
from torch.nn import functional as F

from models.shared_blocks.DecoderBlock import DecoderBlock
from models.shared_blocks.PositionalEncoding import PositionalEncoding

# padding idx is hardcoded
PADDING_IDX = 1

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # why check specifically these two?
        assert config.vocab_size is not None
        assert config.context_length is not None

        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=PADDING_IDX)
        self.position_embedding_table = PositionalEncoding(config)
        self.attn_blocks =    nn.ModuleList([DecoderBlock(config) for _ in range(config.n_blocks)])
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

        self.device = config.device
        
    def forward(self, idx, targets=None):
        # B = batch size
        # T = timestep, or, how many tokens in one instance
        B, T = idx.shape

        
        """
        small note on why embedding is called 'channel' (chatgpt, but i think makes sense):
        The reason "embedding size" can be called a "channel" is because each dimension of an embedding can be thought of as a separate feature or channel.
        Just like each color channel captures different information in an image, each dimension in the embedding captures different aspects of the word's meaning.
        """
        # making every token a vector (their embedding size), therefore the C (channel)
        vocab_embd_output = self.token_embedding_table(idx) # B T C
        # each position is going to be added with a positional encoding
        positional_embd = self.position_embedding_table(T) # T C
        # broadcast operation by pytorch because it's (B, T, C) and (T, C) -> second operand is added to every element in B
        x = vocab_embd_output + positional_embd # B T C

        # forward pass for every block
        for block in self.attn_blocks:
            x = block(x)

        # get logits (vocab sized, not softmaxed)
        logits = self.lm_head(x)
    
        if targets is None:
            loss = None     # yea u can't do this bc what target to compare it to?
        else:
            B, T, C = logits.shape          # each token has it's own logits, to know what token after it is
            
            # reshaping so that we can use this function from pytorch
            # this means loss is batch loss instead of only one instance
            logits = logits.view(B*T, C)    # flattening the batch
            targets = targets.view(B*T)     # original targets dimension is B,T
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx : sequence of tokens
        # generate max_new_tokens times
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.context_length:]) # B T C
            logits = logits[:, -1, :]   # get the last channel of the sequence. B 1 C
            probs = F.softmax(logits, dim=-1)   # softmax on the C
            idx_next = torch.multinomial(probs, num_samples=1)  # get the actual token based on the probs
            idx = torch.cat((idx, idx_next), dim=1) # append the new token to the sequence
        return idx