from models.RNN.RNN import RNN

from utils.data_utils import *
from utils.config import Config

data_path = 'input.txt'

# variables ==========================================================
batch_size = 4
max_iters = 3

# setting up data ====================================================
text, chars, vocab_size, encode, decode = text_chars_vocabsz_enc_dec(data_path)
data = torch.tensor(encode(text))

n = int(0.9*len(chars))
train_data = data[:n]
val_data = data[n:]

# model config =======================================================
config = Config(
    vocab_size=vocab_size,
    context_length=8,
    embedding_size=32,

    # unused, for RNN
    n_attn_heads=None,
    n_attn_blocks=None,
    layer_norm_bias=None,
    dropout=None
)

# MODEL ==============================================================
model = RNN(config)

# training loop ======================================================
for i in range(max_iters):
    xb, yb = get_batch(train_data, config.context_length, batch_size)
    logits, loss = model(xb,yb)