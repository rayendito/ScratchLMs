from utils.Tokenizer import Tokenizer

input_file = 'input.txt'
tokenizer = Tokenizer(input_file, 100)

print(tokenizer('hello world!'))
print(tokenizer.decode(tokenizer('hello world!')[0]))