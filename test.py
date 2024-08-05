from utils.Tokenizer import Tokenizer

input_file = 'input_smaller.txt'
tokenizer = Tokenizer(input_file, 60)


# inputstr = "After BIOS completes some low-level tests of the hardware, particularly whether ornot the installed memory is working correctly, it must boot the operating system storedon one of your devices."

# print(tokenizer('hello 안녕하세요 world!'))
print('hello 안 llo녕하 lkj세요 world!')
print(tokenizer.decode(tokenizer('hello 안 llo녕하 lkj세요 world!')[0]))