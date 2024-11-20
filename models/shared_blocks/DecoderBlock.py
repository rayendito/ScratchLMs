import torch.nn as nn
from .AttentionLayer import AttentionLayer
from .MLP import MLP
from .LayerNorm import LayerNorm

class DecoderBlock(nn.Module):
    def __init__(self, config, cross = False):
        super().__init__()
        self.cross = cross

        # 1. Masked Multi Head Attention
        # meaning that its masked with the tril function so that it cannot attend to the future
        self.norm1 = LayerNorm(config.embedding_size, config.layer_norm_bias)
        self.self_attention = AttentionLayer(config, causal=True, cross=False) # causal is the param that toggles this

        # 2. Cross Attention (GPT does not need this, therefore optional)
        # K and V from some external source, can attend to those two for as much as it likes
        if(self.cross):
            self.norm2 = LayerNorm(config.embedding_size, config.layer_norm_bias)
            self.cross_attention = AttentionLayer(config, causal=False, cross=True) # notice that causal is False
        
        # 3. FFNN
        # 3blue1brown says this is where memory is kept
        self.norm3 = LayerNorm(config.embedding_size, config.layer_norm_bias)
        self.mlp = MLP(config)

    def forward(self, x, cross_attn_key=None, cross_attn_value=None, kv_cache=None):        
        """
        why do we add and normalize?
        - add: so that gradients flow fast
        - normalize: uhhh internal covariate shift or something
            i been reading and it often says so that the training is more stable when the values are at the same range
            or something? not really sure tbh *why* that works

        note: i did normalize on x first then add and normalize it again lol
        this was a mistake lmao but it turns out to be empirically better than the regular add and normalize :V
        so i kept it
        """
        x = self.norm1(x)
        if(kv_cache is None):
            x  = x + self.self_attention(x)
        else:
            x_attn, new_kv = self.self_attention(x, kv_cache=kv_cache)
            x = x + x_attn
        
        if(self.cross):
            assert cross_attn_key and cross_attn_value
            x = self.norm2(x)
            x = x + self.cross_attention(x, cross_attn_key, cross_attn_value)
        
        x = self.norm3(x)
        x = x + self.mlp(x)

        # print("returned x type", type(x))
        if(kv_cache is None):
            return x
        else:
            return x, new_kv