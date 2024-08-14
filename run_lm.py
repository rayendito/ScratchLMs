import torch

from models.GPT.GPT import GPT
from models.RNN.RNN import RNN

from utils.Tokenizer import Tokenizer
from utils.config import GPTConfig, RNNConfig
from utils.model_utils import show_parameter_counts

# hyperparameters ====================================================
batch_size = 4
max_iters = 1000
eval_interval = 100
eval_iters = 200
lr = 1e-5

target_vocab_size = 400

# argparse ====================================================
import argparse
parser = argparse.ArgumentParser(description="Specify the architecture and training corpus")
parser.add_argument('--architecture', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--token_strategy', type=str, default='char')

args = parser.parse_args()
ARCHITECTURE = args.architecture.lower()
LM_DATA_DIR = 'data/mono/'
DATA_PATH = LM_DATA_DIR + args.data_path
TOKEN_STRATEGY = args.token_strategy.lower()

VALID_ARCHITECTURES = ['gpt', 'rnn']
assert ARCHITECTURE in VALID_ARCHITECTURES, f"Invalid architecture '{ARCHITECTURE}'. Choose from {VALID_ARCHITECTURES}."

VALID_TOK_STRAT = ['char', 'byte', 'code_point']
assert TOKEN_STRATEGY in VALID_TOK_STRAT, f"Invalid tokenizer strategy '{TOKEN_STRATEGY}'. Choose from {VALID_TOK_STRAT}."

# estimate_loss function ======================================================
def estimate_loss(model, data_train, data_val):
    out = {}
    splits = {'train' : data_train, 'val' : data_val}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = tokenizer.get_batch(splits[split], config.context_length, batch_size)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# device ====================================================
if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'
# for that day when MPS is actually good lol
# elif(torch.backends.mps.is_available()): # MPS currently is still slower than CPUs?
#     device = 'mps'

if __name__ == "__main__":
    # tokenizer ====================================================
    if(TOKEN_STRATEGY == 'char'):
        tokenizer = Tokenizer(DATA_PATH)
    elif(TOKEN_STRATEGY == 'byte' or TOKEN_STRATEGY == 'code_point'):
        tokenizer = Tokenizer(DATA_PATH,
                              target_vocab_size=target_vocab_size,
                              encoding_level=TOKEN_STRATEGY)
    
    # model ====================================================
    if(ARCHITECTURE == 'gpt'):
        config = GPTConfig(vocab_size=tokenizer.vocab_size, device=device)
        model = GPT(config).to(device)
    elif(ARCHITECTURE == 'rnn'):
        config = RNNConfig(vocab_size=tokenizer.vocab_size, device=device)
        model = RNN(config).to(device)
    show_parameter_counts(model)

    # optimizer ====================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    # data set up ====================================================
    all_data_tokenized = tokenizer.encode_from_file(DATA_PATH)[0]
    train_size = int(0.9*len(all_data_tokenized))
    train_data = all_data_tokenized[:train_size]
    val_data = all_data_tokenized[train_size:]

    # training loop ======================================================
    for i in range(max_iters):
        if(i % eval_interval == 0):
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = tokenizer.get_batch(train_data, config.context_length, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb,yb)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

    # generation =========================================================
    seed = 'We are accounted poor cit'
    seed_encoded = tokenizer(seed).to(device)
    result = model.generate(seed_encoded, 25)
    print(tokenizer.decode(result[0]))