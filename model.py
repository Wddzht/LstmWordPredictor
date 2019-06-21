import tensorflow as tf
import numpy as np


class LSTMPredictor:
    def __init__(self,
                 batch_size,
                 vocab_size,
                 learning_rate=0.001,
                 grad_clip=5,
                 embedding_dim=800,
                 lstm_layer_num=3,
                 embeddings=None,
                 update_embedding=True):
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.update_embedding = update_embedding  # 测试时 update_embedding = False
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.embedding_dim = embedding_dim
        self.lstm_layer_num = lstm_layer_num
        self.classes_num = vocab_size + 1  # todo: See `data.make_vocab()`

        self.input_seqs = tf.placeholder(tf.int32, shape=[None, None], name="input_seqs")
        self.target_seqs = tf.placeholder(tf.int32, shape=[None, None], name="target_seqs")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # build graph
        self.word_embeddings = self.build_embedding_layer(self.input_seqs)
        self.lstm_output, self.final_state = self.lstm_layer(self.word_embeddings)
        self.logits, self.prediction = self.output_layer(self.lstm_output)
        self.loss = self.get_loss(self.logits, self.target_seqs)
        self.optimizer = self.get_optimizer(self.loss)

        self.init_op = tf.global_variables_initializer()

    def build_embedding_layer(self, seqs):
        if self.embeddings is None:
            self.embeddings = random_embedding(self.classes_num, self.embedding_dim)
        with tf.variable_scope("embedding"):
            word_embeddings = tf.Variable(self.embeddings,
                                          dtype=tf.float32,
                                          trainable=self.update_embedding,
                                          name="word_embeddings")
            lookup_embeddings = tf.nn.embedding_lookup(params=word_embeddings, ids=seqs)
        return tf.nn.dropout(lookup_embeddings, self.keep_prob)

    def lstm_layer(self, inputs):
        # todo: units num 是否只能等于 embedding dim?
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_dim)

        # todo:lstm 的 keep_prob 和 embedding 的 keep_prob用了同一个参数
        lstm_drop = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

        '''
        new_cell = MultiRNNCell([get_a_cell() for _ in range(n)])
            得到的 cell 实际也是 RNNCell 的子类,也有 call 方法，它的 state_size 是 n*cell_size
            表示共有n个隐层状态，每个隐层状态的大小为 cell_size
        '''
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_drop for _ in range(self.lstm_layer_num)])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        '''
        (outputs, state) = dynamic_rnn(cell,inputs,sequence_length=None,)
            outputs: 每个cell的输出
            states：表示最终的状态，也就是序列中最后一个cell输出的状态。
            
            一般情况下：
                当 cell 为 GRU,RNN 时，states 的形状为 [batch_size, cell.output_size ]，
                state 就只有一个，原因是GRU将遗忘门和输入门合并成了更新门，cell 不再有细胞状态 cell state
                
                当 cell 为 BasicLSTMCell 时，state 的形状为[2，batch_size, cell.output_size]，
                其中2也对应着 LSTM 中的 cell state 和 hidden state
            https://zhuanlan.zhihu.com/p/43041436
        '''
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state)
        return outputs, state

    def output_layer(self, lstm_output):
        """
        soft_max layer
            input_size: lstm size
            output_size: vocab size
        :param lstm_output:
        :return:
        """
        seq_output = tf.concat(lstm_output, 1)
        x = tf.reshape(seq_output, [-1, self.embedding_dim])
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable(shape=[self.embedding_dim, self.classes_num],
                                        name='w',
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(shape=[self.classes_num],
                                        name='b',
                                        initializer=tf.zeros_initializer(),
                                        dtype=tf.float32)

        logits = tf.matmul(x, softmax_w) + softmax_b
        output = tf.nn.softmax(logits)

        return logits, output

    def get_loss(self, logits, target_seqs):
        target_seqs_ = tf.reshape(target_seqs, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_seqs_)
        loss = tf.reduce_mean(losses)
        return loss

    def get_optimizer(self, loss):
        trainable_var = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_var), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, trainable_var))

        return optimizer


def random_embedding(vocab_size, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding = np.float32(embedding_mat)
    return embedding
