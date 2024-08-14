import torch
from utils.Tokenizer import Tokenizer

input_file = 'data/mono/input.txt'
tokenizer = Tokenizer(input_file)

inputss = [
    'hello world! my name is Jude',
    'But these were days',
]

print(tokenizer(inputss))
print(tokenizer.decode(tokenizer(inputss)))

