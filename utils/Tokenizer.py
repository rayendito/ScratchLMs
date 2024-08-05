import torch

PAD_CHAR='@'

class Tokenizer():
    def __init__(self, input_file,
                  target_vocab_size,
                  encoding_level='char',
                  train_char_coverage=0.8, # char level encoding arg. more info https://github.com/google/sentencepiece/issues/412
                  byte_fallback=False): # char level encoding arg.
        with open(input_file, 'r', encoding='utf8') as f:
            self.text = f.read() # i dont think it's wise to save the whole text in the object
            self.text_utf8 = list(self.text.encode('utf8'))

            self.encoding_level = encoding_level
            self.byte_fallback = byte_fallback
            if(self.encoding_level == 'char'):
                self.train_bpe_char()
            elif(self.encoding_level == 'byte'):
                self.train_bpe_byte(target_vocab_size) # vocab is initialized through calling this function
            elif(self.encoding_level == 'code_point'):
                self.train_bpe_code_point(target_vocab_size, train_char_coverage, byte_fallback)
            else:
                print('INVALID ENCODING LEVEL ARG')
            
            self.vocab_size = len(self.vocab)

    # encode a string
    def __call__(self, input_string):
        if(self.encoding_level == 'char'):
            return torch.tensor([[self.reversed_vocab[c] if c in self.reversed_vocab else 0 for c in input_string]])
        elif(self.encoding_level == 'byte'):
            input_encoded = input_string.encode('utf-8')
            input_tokenized = list(input_encoded)
            for idx, pair in self.vocab.items():
                pair = tuple(pair)
                if(len(pair) >= 2):
                    input_tokenized = self.merge(input_tokenized, pair, idx) 
            return torch.tensor([input_tokenized])
        elif(self.encoding_level == 'code_point'):
            # algorithm complexity is crazy here tho lol
            input_tokenized = []
            while(input_string != ''):
                cut = False
                for tok, i in self.reversed_vocab.items():
                    if input_string.startswith(tok):
                        input_tokenized.append(i)
                        input_string = input_string[len(tok):]
                        cut = True
                if(not(cut)):
                    if(self.byte_fallback):
                        bytestream = list(input_string[0].encode('utf-8'))
                        for b in bytestream:
                            input_tokenized.append(self.reversed_vocab[f'<{str(hex(b))}>'])
                    else:
                        if(input_tokenized and input_tokenized[-1] != self.reversed_vocab['<UNK>']):
                            input_tokenized.append(self.reversed_vocab['<UNK>'])
                    input_string = input_string[1:]
            return torch.tensor([input_tokenized])
    
    # encode but from a file name
    def encode_from_file(self, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            text = f.read()
            return self(text)

    # daydreaming lol
    def train_bpe_char(self):
        chars = sorted(list(set(self.text)))
        self.vocab = {
            0 : '⌬' # unknown character fallback
        }
        for i,ch in enumerate(chars):
            self.vocab[i+1] = ch
        self.reversed_vocab = { v : k for k, v in self.vocab.items() }

    def train_bpe_byte(self, target_vocab_size):
        # hardcoded 256?
        # it's because one byte is maximum 255 (0-255 is 256 things)
        self.vocab = {i : bytes([i]) for i in range(256)}
        n_merges = target_vocab_size - 256
        for i in range(n_merges):
            stats = self.get_pair_stats(self.text_utf8)
            most_freq = max(stats, key=stats.get)
            self.text_utf8 = self.merge(self.text_utf8, most_freq, 256+i)
            self.vocab[256+i] = self.vocab[most_freq[0]] + self.vocab[most_freq[1]]
        self.reversed_vocab = {v:k for k,v in self.vocab.items()}

    def train_bpe_code_point(self, target_vocab_size, train_char_coverage, byte_fallback):
        train_len = round(len(self.text) * train_char_coverage)
        self.reversed_vocab = {
            '<UNK>' : 0,
        }
        
        # add to the vocab first the individual characters that exist?
        if byte_fallback:
            # add the byte fallback tokens to vocab
            for i in range(256):
                self.reversed_vocab[f'<{str(hex(i))}>'] = len(self.reversed_vocab)
        
        for c in self.text[:train_len]:
            if(c not in self.reversed_vocab):
                self.reversed_vocab[c] = len(self.reversed_vocab)
        
        # tokenizing result variable with the available
        c_points = [self.reversed_vocab[self.text[i]] for i in range(train_len)]

        # completing the tokenizer based on whether there's byte fallback
        if byte_fallback:
            uncovered_bytes = list(self.text[train_len:].encode('utf-8'))
            for b in uncovered_bytes:
                c_points.append(self.reversed_vocab[f'<{str(hex(b))}>'])
        else:
            c_points.append(self.reversed_vocab['<UNK>'])

        # training the vocab, adding merged vocab
        self.vocab = {v : k for k, v in self.reversed_vocab.items()}
        begin_idx = len(self.reversed_vocab)
        add_vocab = target_vocab_size - begin_idx

        if(add_vocab > 0):
            for i in range(add_vocab):
                stats = self.get_pair_stats(c_points)
                if(len(stats) != 0):
                    freq_pair = max(stats, key=stats.get)
                    c_points = self.merge(c_points, freq_pair, begin_idx+i)

                    new_word = self.vocab[freq_pair[0]] + self.vocab[freq_pair[1]]
                    self.reversed_vocab[new_word] = begin_idx+i
                    self.vocab[begin_idx+i] = new_word
                else:
                    print("maximum amount of merges reached. stopping merge.")
                    break
        
        #sorting the reversed vocab so that it can be used to decode greedily
        self.reversed_vocab = dict(sorted(self.reversed_vocab.items(), key=lambda item: len(item[0]), reverse=True))

    def decode(self, ids):
        ids = ids.tolist()
        if(self.encoding_level == 'char'):
            return ''.join([self.vocab[i] for i in ids])
        elif(self.encoding_level == 'byte'):
            bytestream = b"".join([self.vocab[i] for i in ids]) # ids are indices
            # 128 to 255 would return an error. not a valid utf hex
            # but 256 and beyond would always be a combination of 0-127 or >255 no?
            # so 128-255 should never be accessed?
            return bytestream.decode('utf-8', errors='replace')
        elif(self.encoding_level == 'code_point'):
            res = []
            i = 0
            while i < len(ids):
                if(not(self.vocab[ids[i]].startswith('<0x'))):
                    res.append(self.vocab[ids[i]])
                    i += 1
                else:
                    stream_len = 1
                    while(i+stream_len < len(ids) and self.vocab[ids[i+stream_len]].startswith('<0x')):
                        stream_len += 1
                    if(self.byte_fallback):
                        todecode = [self.vocab[d] for d in ids[i:i+stream_len]]
                        bytestream = self.__get_bytestream(todecode)
                        res.append(bytestream.decode('utf-8'))
                    i += stream_len
            return "".join(res)
    
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


    def __get_bytestream(self, hex_list):
        # Process each string in the list and combine the bytes
        return b''.join(self.__extract_byte(hex_str) for hex_str in hex_list)

    def __extract_byte(self, hex_string):
        # Remove the surrounding '<>' characters and '0x' prefix
        hex_string = hex_string.strip('<>').lstrip('0x')
        # Convert hex string to a single byte
        return bytes([int(hex_string, 16)])

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