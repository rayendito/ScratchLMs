from tqdm import tqdm
from models.BigramLanguageModel import BigramLanguageModel
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
epoch = 10000
lr = 1e-3

# setting up data ====================================================
text, chars, vocab_size, encode, decode = text_chars_vocabsz_enc_dec(data_path)
data = torch.tensor(encode(text))

n = int(0.9*len(chars))
train_data = data[:n]
val_data = data[n:]

# model and optimizers ===============================================
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)

# training loop ======================================================
for _ in tqdm(range(epoch)):
    xb, yb = get_batch(train_data, block_size, batch_size)
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

print('last epoch loss is', loss.item())

# inference ==========================================================
seed = 'dear'
seed_encoded = torch.tensor([encode(seed)]).to(device)
result = model.generate(seed_encoded, 100)
print(decode(result[0].tolist()))