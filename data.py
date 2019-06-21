import pickle


def make_train_data(data):
    for batch in data:
        target = []
        for seq in batch:
            seq_target = seq[1:]
            seq_target.append(0)  # 末尾用0补齐
            target.append(seq_target)

        yield batch, target


def make_vocab(string, vocab_path, min_count=-1):
    """
    make vocab
    :param min_count:
    :param vocab_path:
    :param string:
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


def read_str_data(data_path, vocab, len_seqs=40, n_seqs=32):
    input_file = open(data_path, encoding='utf8')
    data = input_file.readlines()

    input_str = ''
    for line in data:
        line = line.replace('\n', '')
        line = line.replace('）', '')
        line = line.replace('（', '')
        line = line.split(':')[1]
        input_str += line

    batch_size = n_seqs * len_seqs
    n_batches = int(len(input_str) / batch_size)
    max_index = batch_size * n_batches
    output_seqs = []
    index = 0

    while index < max_index:
        batches = []
        while len(batches) < n_seqs:
            current_seq = []
            while len(current_seq) < len_seqs:
                current_seq.append(word2id(input_str[index], vocab))
                index += 1
            batches.append(current_seq)
        output_seqs.append(batches)

    print("data length: {}\tbatch num: {}".format(len(input_str), n_batches))
    return output_seqs
