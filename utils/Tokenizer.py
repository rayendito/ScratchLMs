import torch

PAD_CHAR='@'

class Tokenizer():
    def __init__(self, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            text = f.read() # i dont think it's wise to save the whole text in the object
            self.chars = [PAD_CHAR] + sorted(list(set(text)))
            self.vocab_size = len(self.chars)
            self.stoi = { ch:i for i, ch in enumerate(self.chars) }
            self.itos = { i:ch for i, ch in enumerate(self.chars) }
            self.data = torch.tensor(self(text))

    def __call__(self, input_string, context_length = 8, padding = False):
        if padding and len(input_string) < context_length:
            input_string = PAD_CHAR*(context_length - len(input_string)) + input_string
        return [self.stoi[c] for c in input_string]

    def decode(self, token_list):
        return ''.join([self.itos[idx] for idx in token_list])
    
    @staticmethod
    def get_batch(data, block_size, batch_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
