import os
import torch

UNK_TOK = '<UNK>'
PAD_TOK = '<PAD>'
SRC_TOK = '<SRC>'
TGT_TOK = '<TGT>'
END_TOK = '<END>'

class Tokenizer():
    def __init__(self, input_file,
                  target_vocab_size=None,
                  encoding_level='char',
                  train_char_coverage=0.8, # char level encoding arg. more info https://github.com/google/sentencepiece/issues/412
                  byte_fallback=False): # char level encoding arg.
        with open(input_file, 'r', encoding='utf8') as f:
            self.text = f.read() # i dont think it's wise to save the whole text in the object
            self.text_utf8 = list(self.text.encode('utf8'))            
            self.byte_fallback = byte_fallback


            self.encoding_level = encoding_level
            if(self.encoding_level == 'char'):
                self.train_char()
            elif(self.encoding_level == 'byte'):
                self.train_bpe_byte(target_vocab_size) # vocab is initialized through calling this function
            elif(self.encoding_level == 'code_point'):
                self.train_bpe_code_point(target_vocab_size, train_char_coverage, byte_fallback)
            else:
                print('INVALID ENCODING LEVEL ARG')
            
            self.vocab_size = len(self.vocab)

    # encode a string
    def __call__(self, inputs, pad=True):
        if(isinstance(inputs, str)):
            inputs = [inputs]
        
        if(self.encoding_level == 'char'):
            ids = [self.encode_char(inp) for inp in inputs]
        elif(self.encoding_level == 'byte'):
            ids = [self.encode_bpe_byte(inp) for inp in inputs]
        elif(self.encoding_level == 'code_point'):
            ids = [self.encode_bpe_code_point(inp) for inp in inputs]

        if(pad):
            ids = self.pad_batch(ids)

        return torch.tensor(ids)
    
    # ===============================================================================================
    # ENCODING TEXTFILES
    # ===============================================================================================
    def encode_from_mono_file(self, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            text = f.read()
            return self(text)
        
    def encode_from_para_dir(self, para_dir):
        src_path = os.path.join(para_dir, 'src.txt')
        tgt_path = os.path.join(para_dir, 'tgt.txt')
        assert os.path.isfile(src_path) and os.path.isfile(tgt_path), f"both src.txt and tgt.txt need to exist in {para_dir}"

        with open(src_path, 'r') as src_file:
            src_lines = src_file.readlines()
            src_lines = [line.strip() for line in src_lines]

        with open(tgt_path, 'r') as tgt_file:
            tgt_lines = tgt_file.readlines()
            tgt_lines = [line.strip() for line in tgt_lines]
        
        assert len(src_lines) == len(tgt_lines), f"unequal src and tgt lens: {len(src_lines)} {len(tgt_lines)}"

        # encoding with added tokens
        ids = []
        for i in range(len(src_lines)):
            ids.append([self.reversed_vocab[SRC_TOK]] + 
                            self(src_lines[i], pad=False)[0].tolist() +
                            [self.reversed_vocab[TGT_TOK]] +
                            self(tgt_lines[i], pad=False)[0].tolist() +
                            [self.reversed_vocab[END_TOK]])
        ids = self.pad_batch(ids)
        return torch.tensor(ids)


    # ===============================================================================================
    # TRAINING
    # ===============================================================================================
    def train_char(self):
        chars = sorted(list(set(self.text)))
        self.vocab = {
            0 : UNK_TOK, # unknown character fallback
            1 : PAD_TOK,
            2 : SRC_TOK,
            3 : TGT_TOK,
            4 : END_TOK,
        }
                                                
        v_len = len(self.vocab)
        for i,ch in enumerate(chars):
            self.vocab[i+v_len] = ch
        self.reversed_vocab = { v : k for k, v in self.vocab.items() }

    def train_bpe_byte(self, target_vocab_size):
        # hardcoded 256?
        # it's because one byte is maximum 255 (0-255 is 256 things)
        assert target_vocab_size != None
        self.vocab = {i : bytes([i]) for i in range(256)}
        n_merges = target_vocab_size - 256
        for i in range(n_merges):
            stats = self.get_pair_stats(self.text_utf8)
            most_freq = max(stats, key=stats.get)
            self.text_utf8 = self.merge(self.text_utf8, most_freq, 256+i)
            self.vocab[256+i] = self.vocab[most_freq[0]] + self.vocab[most_freq[1]]
        self.reversed_vocab = {v:k for k,v in self.vocab.items()}

    def train_bpe_code_point(self, target_vocab_size, train_char_coverage, byte_fallback):
        assert target_vocab_size != None
        train_len = round(len(self.text) * train_char_coverage)
        self.reversed_vocab = {
            UNK_TOK : 0,
            PAD_TOK : 1,
            SRC_TOK : 2,
            TGT_TOK : 3,
            END_TOK : 4,
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
            c_points.append(self.reversed_vocab[UNK_TOK])

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
    
    # ===============================================================================================
    # ENCODING
    # ===============================================================================================
    def encode_char(self, input_string):
        return [self.reversed_vocab[c] if c in self.reversed_vocab else 0 for c in input_string]
    
    def encode_bpe_byte(self, input_string):
        input_encoded = input_string.encode('utf-8')
        input_tokenized = list(input_encoded)
        for idx, pair in self.vocab.items():
            pair = tuple(pair)
            if(len(pair) >= 2):
                input_tokenized = self.merge(input_tokenized, pair, idx) 
        return input_tokenized
    
    #TODO: make better? (complexity is crazy here lol)
    def encode_bpe_code_point(self, input_string):
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
                    if(input_tokenized and input_tokenized[-1] != self.reversed_vocab[UNK_TOK]):
                        input_tokenized.append(self.reversed_vocab[UNK_TOK])
                input_string = input_string[1:]
        return input_tokenized
    
    # ===============================================================================================
    # DECODING 
    # ===============================================================================================
    # support delete_special currently only for char encoding level
    def decode(self, ids, delete_special = True):
        if(ids.ndim == 1):
            ids = [ids.tolist()]
        else:
            ids = ids.tolist()
        
        if(self.encoding_level == 'char' and delete_special):
            ids = [
                [x for x in i if 4 < x] for i in ids
            ]

        if(self.encoding_level == 'char'):
            return [self.decode_char(i) for i in ids]
        elif(self.encoding_level == 'byte'):
            return [self.decode_bpe_byte(i) for i in ids]
        elif(self.encoding_level == 'code_point'):
            return [self.decode_bpe_code_point(i) for i in ids]
    
    def decode_char(self, ids):
        return ''.join([self.vocab[i] for i in ids])
    
    def decode_bpe_byte(self, ids):
        # 128 to 255 would return an error. not a valid utf hex
        # but 256 and beyond would always be a combination of 0-127 or >255 no?
        # so 128-255 should never be accessed?
        bytestream = b"".join([self.vocab[i] for i in ids]) # ids are indices
        return bytestream.decode('utf-8', errors='replace')
    def decode_bpe_code_point(self, ids):
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

    # ===============================================================================================
    # UTILS
    # ===============================================================================================
    def pad_batch(self, ids):
        longest = self.__get_longest_strlen_in_batch(ids)
        ids = [
            [self.reversed_vocab[PAD_TOK]] * (longest - len(i)) + i for i in ids
        ]
        return ids
    
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

    def __get_longest_strlen_in_batch(self, batch):
        longest = -99
        for st in batch:
            if len(st) > longest:
                longest = len(st)
        return longest

    # ===============================================================================================
    # STATICS
    # ===============================================================================================
    @staticmethod
    def get_batch_from_mono(data, block_size, batch_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
    
    @staticmethod
    def get_batch_from_para(data, batch_size):
        ix = torch.randint(len(data), (batch_size,))
        x = torch.stack([data[i][:-1] for i in ix])
        y = torch.stack([data[i][1:] for i in ix])
        return x, y