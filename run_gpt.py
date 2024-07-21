# from modules.attention_model import nanoGPT
from models.GPT.GPT import GPT
from utils.data_utils import *
from utils.config import Config

torch.manual_seed(1337)

# variables ==========================================================
data_path = 'input.txt'

if(torch.cuda.is_available()):
    device = 'cuda'
# MPS currently is still slower than CPUs?
# elif(torch.backends.mps.is_available()):
#     device = 'mps'
else:
    device = 'cpu'

# setting up data ====================================================
text, chars, vocab_size, encode, decode = text_chars_vocabsz_enc_dec(data_path)
data = torch.tensor(encode(text))

n = int(0.9*len(chars))
train_data = data[:n]
val_data = data[n:]

# hyperparameters ====================================================
batch_size = 4
max_iters = 1500
eval_interval = 100
eval_iters = 200
lr = 3e-4

# model config =======================================================
config = Config(
    vocab_size=vocab_size,
    context_length=8, # ex block_size
    embedding_size=32,
    n_attn_heads=4,
    n_blocks=5,
    layer_norm_bias=False,
    dropout=0
)

# model and optimizers ===============================================
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)

# training loop ======================================================
def estimate_loss(model, data_train, data_val):
    out = {}
    splits = {'train' : data_train, 'val' : data_val}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(splits[split], config.context_length, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# for it in tqdm(range(max_iters)):
for it in range(max_iters):
    if(it % eval_interval == 0):
        losses = estimate_loss(model, train_data, val_data)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch(train_data, config.context_length, batch_size)
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    
    loss.backward()
    optimizer.step()
    
    optimizer.zero_grad(set_to_none = True)

# inference ==========================================================
# todo: padding to context length function
seed = 'Becometh'
seed_encoded = torch.tensor([encode(seed)]).to(device)
result = model.generate(seed_encoded, 100)
print(decode(result[0].tolist()))

