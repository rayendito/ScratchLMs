from tqdm import tqdm
from models.attention_model import nanoGPT
from utils.data_utils import *

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

# hyperparameters ====================================================
batch_size = 4
block_size = 8
max_iters = 1500
eval_interval = 100
eval_iters = 200
lr = 1e-4


# setting up data ====================================================
text, chars, vocab_size, encode, decode = text_chars_vocabsz_enc_dec(data_path)
data = torch.tensor(encode(text))

n = int(0.9*len(chars))
train_data = data[:n]
val_data = data[n:]

# model and optimizers ===============================================
model = nanoGPT(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)

# training loop ======================================================
def estimate_loss(model, data_train, data_val):
    out = {}
    splits = {'train' : data_train, 'val' : data_val}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(splits[split], block_size, batch_size)
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

    xb, yb = get_batch(train_data, block_size, batch_size)
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# inference ==========================================================
# todo: padding to context length function
seed = 'Becometh'
seed_encoded = torch.tensor([encode(seed)]).to(device)
result = model.generate(seed_encoded, 100)
print(decode(result[0].tolist()))

