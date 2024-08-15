import torch
from utils.Tokenizer import Tokenizer

input_file = 'data/mono/input.txt'
tokenizer = Tokenizer(input_file)

# inputss = [
#     'hello world! my name is Jude',
#     'But these were days',
# ]

# print(tokenizer(inputss))
# print(tokenizer.decode(tokenizer(inputss)))


para_dir = 'data/para/spa_mt'
para_tokenized = tokenizer.encode_from_para_dir(para_dir)
# print(para_tokenized)

aaa = tokenizer.decode(para_tokenized, delete_special=False)
for a in aaa:
    print(a)
# print(tokenizer.decode(para_tokenized, delete_special=False))