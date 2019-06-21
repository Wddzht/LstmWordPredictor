# coding=utf-8
import argparse
import os
import time

import tensorflow as tf
import numpy as np

import data
from model import LSTMPredictor

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

# hyper parameters
parser = argparse.ArgumentParser(description='LSTM Predictor task')
parser.add_argument('--data_path', type=str, default=r'E:\_Python\Data\word_predictor\cut_poetry.txt',
                    help='train data source')
parser.add_argument('--vocab_path', type=str, default=r'E:\_Python\Data\word_predictor\vocab\cut_poetry.vocab',
                    help='vocab path')
parser.add_argument('--output_path', type=str, default=r'E:\_Python\Data\word_predictor\output_save',
                    help='output path')

parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--model_path', type=str, default='#', help='model path for test or demo')

parser.add_argument('--batch_size', type=int, default=32, help='#sample of each mini batch')
parser.add_argument('--epoch', type=int, default=45, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=700, help='#dim of hidden state')  # todo: 是否需要?
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--keep_prob', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--lstm_layer_num', type=int, default=3, help='lstm layer num')

parser.add_argument('--embedding_dim', type=int, default=128, help='embedding size')
parser.add_argument('--update_embedding', type=bool, default=True, help='update embedding during training True/False')
parser.add_argument('--pretrained_embedding', type=str, default='random',
                    help='pretrained embedding path or init it randomly')

args = parser.parse_args()

if args.mode == 'train':
    timestamp = str(int(time.time()))
    output_path = os.path.join(args.output_path, timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ckpt_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    model_path = os.path.join(ckpt_path, "model")

    vocab = data.read_vocab(args.vocab_path)
    encoded_seqs = data.read_str_data(args.data_path, vocab, len_seqs=40, n_seqs=32)

    model = LSTMPredictor(batch_size=args.batch_size,
                          vocab_size=len(vocab),
                          learning_rate=args.lr,
                          grad_clip=args.clip,
                          embedding_dim=args.embedding_dim,
                          lstm_layer_num=args.lstm_layer_num,
                          embeddings=None,
                          update_embedding=args.update_embedding)

    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(model.init_op)
        new_state = sess.run(model.initial_state)
        counter = 0
        for epoch in range(args.epoch):
            batches = data.make_train_data(encoded_seqs)
            for input_seqs, target_seqs in batches:
                counter += 1
                start = time.time()
                feed_dict = {model.input_seqs: input_seqs,
                             model.target_seqs: target_seqs,
                             model.keep_prob: args.keep_prob,
                             model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer],
                                                    feed_dict=feed_dict)

                end = time.time()

                if counter % 200 == 0:
                    print('epoch: {}/{} '.format(epoch + 1, args.epoch),
                          'step: {} '.format(counter),
                          'err rate: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if counter % 1000 == 0:
                    saver.save(sess, model_path, counter)

        saver.save(sess, model_path, counter)


elif args.mode == 'demo':
    vocab = data.read_vocab(args.vocab_path)


    def get_word_randomly(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        index = int(np.searchsorted(t, np.random.rand(1) * s))
        while id == 0:
            index = int(np.searchsorted(t, np.random.rand(1) * s))  # todo:
        return data.id2words(index, vocab)


    def get_word_max_conf(weights):
        weights = list(weights[0][1:])  # id=0 不考虑
        w = sorted(weights)
        value = w[np.random.randint(len(w) - 10, len(w))]  # len(w)>=10
        index = weights.index(value)
        return data.id2words(index + 1, vocab)


    def sample(checkpoint, max_len):
        model = LSTMPredictor(batch_size=1,  #
                              vocab_size=len(vocab),
                              embedding_dim=args.embedding_dim,
                              lstm_layer_num=args.lstm_layer_num,
                              update_embedding=False)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            new_state_ = sess.run(model.initial_state)

            while True:
                print('please input a char or a string:')
                input_str = input()
                if input_str == 'exit':
                    break

                for c in input_str:
                    input_car = np.zeros((1, 1))
                    input_car[0, 0] = data.word2id(c, vocab)

                    feed_dict_ = {model.input_seqs: input_car,
                                  model.keep_prob: 1.,
                                  model.initial_state: new_state_}
                    prediction, new_state_ = sess.run([model.prediction, model.final_state],
                                                      feed_dict=feed_dict_)

                word = get_word_max_conf(prediction)
                # word = data.id2word[np.argmax(prediction)]

                output_str = input_str
                while len(output_str) <= max_len:
                    output_str += str(word)
                    input_car = np.zeros((1, 1))
                    input_car[0, 0] = data.word2id(word, vocab)

                    feed_dict_ = {model.input_seqs: input_car,
                                  model.keep_prob: 1.,
                                  model.initial_state: new_state_}
                    [prediction, new_state_] = sess.run([model.prediction, model.final_state],
                                                        feed_dict=feed_dict_)
                    word = get_word_max_conf(prediction)
                    # word = data.id2word[np.argmax(prediction)]

                print(output_str)


    ckpt_file = tf.train.latest_checkpoint(args.model_path)
    sample(ckpt_file, 50)
