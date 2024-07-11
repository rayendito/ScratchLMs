import torch

def text_chars_vocabsz_enc_dec(path):
    with open('input.txt', 'r', encoding='utf8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = { ch:i for i, ch in enumerate(chars) }
        itos = { i:ch for i, ch in enumerate(chars) }

        encode = lambda s: [stoi[c] for c in s]
        decode = lambda i: ''.join([itos[idx] for idx in i])

        return text, chars, vocab_size, encode, decode

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

