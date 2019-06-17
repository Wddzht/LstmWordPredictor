from data import make_vocab

input_file = open(r'E:\_Python\Data\word_predictor\cut_poetry.txt', encoding='utf8')
input_lines = input_file.readlines()
input_file.close()

input_str = ''

for line in input_lines:
    line = line.replace('，', ',')
    line = line.replace('。', '.')
    line = line.replace('\n', '')
    line = line.replace('\t', '')
    input_str += line

w_c, del_c = make_vocab(input_str, r'E:\_Python\Data\word_predictor\vocab\cut_poetry.vocab')
print(w_c)
print(del_c)
