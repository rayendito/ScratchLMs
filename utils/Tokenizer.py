import torch

PAD_CHAR='@'

class Tokenizer():
    def __init__(self, input_file, target_vocab_size):
        with open(input_file, 'r', encoding='utf8') as f:
            text = f.read() # i dont think it's wise to save the whole text in the object
            text_utf8 = text.encode('utf8')
            self.text_utf8 = list(text_utf8)

            self.vocab_size = target_vocab_size
            self.train_bpe(target_vocab_size) # vocab is initialized through calling this function

    # encode a string
    def __call__(self, input_string):
        input_encoded = input_string.encode('utf-8')
        input_tokenized = list(input_encoded)
        for idx, pair in self.vocab.items():
            pair = tuple(pair)
            if(len(pair) >= 2):
                input_tokenized = self.merge(input_tokenized, pair, idx) 
        return torch.tensor([input_tokenized])
    
    # encode but from a file name
    def encode_from_file(self, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            text = f.read()
            return self(text)

    def train_bpe(self, target_vocab_size):
        # hardcoded 256?
        # it's because one byte is maximum 255 (0-255 is 256 things)
        self.vocab = {i : bytes([i]) for i in range(256)}
        n_merges = target_vocab_size - 256
        for i in range(n_merges):
            stats = self.get_pair_stats(self.text_utf8)
            most_freq = max(stats, key=stats.get)
            self.text_utf8 = self.merge(self.text_utf8, most_freq, 256+i)
            self.vocab[256+i] = self.vocab[most_freq[0]] + self.vocab[most_freq[1]]

    def decode(self, ids):
        bytestream = b"".join([self.vocab[i] for i in ids.tolist()]) # ids are indices
        # 128 to 255 would return an error. not a valid utf hex
        # but 256 and beyond would always be a combination of 0-127 or >255 no?
        # so 128-255 should never be accessed?
        return bytestream.decode('utf-8', errors='replace')
    
    def get_pair_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, freq_pair, idx):
        # ids is a list of int-ified bytes
        # freq_pair is a pair to be turned into idx
        # idx is a new id
        paired = zip(ids, ids[1:])
        for i, pair in reversed(list(enumerate(paired))):
            if pair == freq_pair:
                ids[i] = idx
                del ids[i+1]
        return ids

    @staticmethod
    def get_batch(data, block_size, batch_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    
    # OLD call dunder (char level)
    # def __call__(self, input_string, context_length = 8, padding = False):
    #     if padding and len(input_string) < context_length:
    #         input_string = PAD_CHAR*(context_length - len(input_string)) + input_string
    #     return [self.stoi[c] for c in input_string]