import torch
import os

from models.GPT import GPT
from models.RNN import RNN

from utils.Tokenizer import Tokenizer
from utils.config import GPTConfig, RNNConfig
from utils.model_utils import show_parameter_counts

torch.manual_seed(1337)

# hyperparameters ====================================================
batch_size = 4
max_iters = 300
eval_interval = 100
eval_iters = 200
lr = 1e-5

target_vocab_size = 400

# argparse ====================================================
import argparse
parser = argparse.ArgumentParser(description="Specify the architecture and training corpus")
parser.add_argument('--architecture', type=str, required=True)
parser.add_argument('--mono_data_path', type=str, required=True)
parser.add_argument('--para_data_dir', type=str, default=None)
parser.add_argument('--token_strategy', type=str, default='char')

args = parser.parse_args()
ARCHITECTURE = args.architecture.lower()
MONO_DATA_PATH = f'data/mono/{args.mono_data_path}'
PARA_DATA_DIR = f'data/para/{args.para_data_dir}'
TOKEN_STRATEGY = args.token_strategy.lower()

VALID_ARCHITECTURES = ['gpt', 'rnn']
assert ARCHITECTURE in VALID_ARCHITECTURES, f"Invalid architecture '{ARCHITECTURE}'. Choose from {VALID_ARCHITECTURES}."

VALID_TOK_STRAT = ['char', 'byte', 'code_point']
assert TOKEN_STRATEGY in VALID_TOK_STRAT, f"Invalid tokenizer strategy '{TOKEN_STRATEGY}'. Choose from {VALID_TOK_STRAT}."

# estimate_loss function ======================================================
def estimate_loss(model, data_train, data_val, source='mono'):
    out = {}
    splits = {'train' : data_train, 'val' : data_val}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if(source == 'mono'):
                X, Y = tokenizer.get_batch_from_mono(splits[split], config.context_length, batch_size)
            elif(source == 'para'):
                X, Y = tokenizer.get_batch_from_para(splits[split], batch_size)
            else:
                raise ValueError(f"{source} is not a valid source")
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
        tokenizer = Tokenizer(MONO_DATA_PATH)
    elif(TOKEN_STRATEGY == 'byte' or TOKEN_STRATEGY == 'code_point'):
        tokenizer = Tokenizer(MONO_DATA_PATH,
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

    # optimizer ========================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    # data set up ======================================================
    TRAIN_PORTION = 0.9
    
    # mono ====================
    mono_data_tokenized = tokenizer.encode_from_mono_file(MONO_DATA_PATH)[0]
    mono_train_size = int(TRAIN_PORTION*len(mono_data_tokenized))
    train_data = mono_data_tokenized[:mono_train_size]
    val_data = mono_data_tokenized[mono_train_size:]

    # para ====================
    if(os.path.isdir(PARA_DATA_DIR)):
        para_data_tokenized = tokenizer.encode_from_para_dir(PARA_DATA_DIR)
        para_train_size = int(TRAIN_PORTION*len(para_data_tokenized))
        para_train_data = para_data_tokenized[:para_train_size]
        para_val_data = para_data_tokenized[para_train_size:]

    # # training loop mono =============================================
    print("BEGINNING LANGUAGE MODEL TRAINING")
    for i in range(max_iters):
        if(i % eval_interval == 0):
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = tokenizer.get_batch_from_mono(train_data, config.context_length, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb,yb)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

    # # training loop parallel ===========================================
    if(os.path.isdir(PARA_DATA_DIR)):
        print("BEGINNING PARALLEL DATA FINETUNING")
        for i in range(max_iters):
            if(i % eval_interval == 0):
                losses = estimate_loss(model, para_train_data, para_val_data, source='para')
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = tokenizer.get_batch_from_para(para_train_data, batch_size)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb,yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)

    # generation ======================================================
    seed = [
        'We are accounted poor citizens',
        'You are all resolved rath',
    ]

    seed_encoded = tokenizer(seed).to(device)
    result = model.generate(seed_encoded, 25)
    result = tokenizer.decode(result)
    print(result[0])
    print(result[1])


