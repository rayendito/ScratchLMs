import torch

PAD_CHAR='@'

def text_chars_vocabsz_enc_dec(path):
    with open(path, 'r', encoding='utf8') as f:
        text = f.read()
        chars = [PAD_CHAR] + sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = { ch:i for i, ch in enumerate(chars) }
        itos = { i:ch for i, ch in enumerate(chars) }


        def encode(input_string, context_length = 8, pad = True):
            if pad and len(input_string) < context_length:
                input_string = PAD_CHAR*(context_length - len(input_string)) + input_string
            return [stoi[c] for c in input_string]

        decode = lambda i: ''.join([itos[idx] for idx in i])

        return text, chars, vocab_size, encode, decode

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

