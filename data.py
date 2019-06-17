import numpy as np
import pickle


def get_batches(arr, n_seqs=32, n_steps=40):
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]

    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def make_vocab(string, vocab_path, min_count=-1):
    """
    make vocab
    :param min_count:
    :param vocab_path:
    :param string:
    :param vocab: {'word':id,...}
    :return:
    """
    vocab = {}
    del_count = 0
    for ch in string:
        if ch not in vocab:
            vocab[ch] = 1
        else:
            vocab[ch] += 1

    if min_count > 0:
        low_freq_words = []
        for word, count in vocab.items():
            if count <= 1:
                low_freq_words.append(word)
        for word in low_freq_words:
            del vocab[word]
            del_count += 1

    next_index = 1
    for word in vocab.keys():
        vocab[word] = next_index
        next_index += 1

    '''
    len(vocab)=len(0,1,2,3,...)=word count+1    0 indicate Unknown
    so embedding size = len(vocab)+1
    else raise InvalidArgumentError：
        indices[x,n] = ? is not in [0, ?) [[node embedding / lookup_embeddings]]
    '''
    vocab[0] = 'U'  # todo:

    with open(vocab_path, 'wb') as fw:
        pickle.dump(vocab, fw)
    return next_index - 1, del_count


def read_vocab(vocab_path):
    with open(vocab_path, 'rb') as fr:
        vocab = pickle.load(fr)
    print('vocab_size:', len(vocab))
    return vocab


def id2words(id, vocab):
    for word, value in vocab.items():
        if value == id:
            return word
    raise ValueError('id2words')


def word2id(word, vocab):
    if word in vocab:
        return vocab[word]
    else:
        return 0


def read_str_data(data_path, vocab):
    input_file = open(data_path, encoding='utf8')
    data = input_file.readlines()

    input_str = ''
    for line in data:
        line = line.replace('\n', '')
        input_str += line

    input_seqs = np.array([], dtype=np.int)
    for ch in input_str:
        input_seqs = np.append(input_seqs, word2id(ch, vocab))
    return input_seqs
