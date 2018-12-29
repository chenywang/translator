# -*- coding:utf-8 -*-
# @Author : Michael-Wang

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import dynamic_rnn

from config import HIDDEN_SIZE, NUM_LAYERS, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, KEEP_PROB, \
    MAX_GRAD_NORM, SOS_ID, EOS_ID, SHARE_EMB_AND_SOFTMAX


class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self, mode):
        assert mode.lower() in ['train', 'predict']
        self.mode = mode.lower()

        self.src_size = None
        self.trg_size = None
        self.src_input = None
        self.trg_input = None
        self.trg_output = None
        self.enc_outputs = None
        self.enc_state = None
        self.train_op = None
        self.cost_op = None
        self.target_id_list = None
        self.softmax_weight = None
        self.softmax_bias = None
        self.src_embedding = None
        self.enc_cell = None
        self.trg_embedding = None
        self.dec_cell = None
        self.build_model()

    def build_model(self):
        self.init_placeholder()
        self.build_encoder()
        self.build_decoder()

    def init_placeholder(self):
        if self.mode == 'predict':
            self.src_size = tf.placeholder(shape=[None, ], dtype=tf.int32)
            self.src_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
        elif self.mode == 'train':
            # self.src_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
            # self.trg_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
            # self.trg_output = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.src_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.trg_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.trg_output = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.src_size = tf.placeholder(shape=[None, ], dtype=tf.int32)
            self.trg_size = tf.placeholder(shape=[None, ], dtype=tf.int32)

    def build_encoder(self):
        with tf.variable_scope('encoder'):
            # 生成embedding层
            self.src_embedding = tf.get_variable(
                "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])

            # 将输入单词编号转为词向量。
            src_emb = tf.nn.embedding_lookup(self.src_embedding, self.src_input)
            self.enc_cell = rnn.MultiRNNCell(
                [rnn.LSTMCell(HIDDEN_SIZE, name='LSTM_{}'.format(_)) for _ in range(NUM_LAYERS)])
            src_emb = tf.nn.dropout(src_emb, KEEP_PROB)

            # 使用dynamic_rnn构造编码器。
            # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
            # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类
            # 张量的tuple，每个LSTMStateTuple对应编码器中的一层。
            # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size,
            # max_time, HIDDEN_SIZE]。Seq2Seq模型中不需要用到enc_outputs，而
            # 后面介绍的attention模型会用到它。
            self.enc_outputs, self.enc_state = dynamic_rnn(
                self.enc_cell, src_emb, self.src_size, dtype=tf.float32)

    def build_decoder(self):
        with tf.variable_scope('decoder'):
            self.trg_embedding = tf.get_variable("trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

            # 定义softmax层的变量
            if SHARE_EMB_AND_SOFTMAX:
                self.softmax_weight = tf.transpose(self.trg_embedding)
            else:
                self.softmax_weight = tf.get_variable(
                    "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
            self.softmax_bias = tf.get_variable(
                "softmax_bias", [TRG_VOCAB_SIZE])

            self.dec_cell = rnn.MultiRNNCell(
                [rnn.LSTMCell(HIDDEN_SIZE, name='LSTM_Go_{}'.format(_)) for _ in range(NUM_LAYERS)])

            if self.mode == 'predict':
                self.build_inference_decoder()
            elif self.mode == 'train':
                # self.build_inference_decoder()
                self.build_train_decoder()

    def build_train_decoder(self):
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, self.trg_input)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 使用dyanmic_rnn构造解码器。
        # 解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步
        # 顶层LSTM的输出。dec_outputs的维度是 [batch_size, max_time, HIDDEN_SIZE]。
        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        dec_outputs, _ = dynamic_rnn(self.dec_cell, trg_emb, self.trg_size, initial_state=self.enc_state)

        # 计算解码器每一步的log perplexity。这一步与语言模型代码相同。
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.trg_output, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰
        # 模型的训练。
        label_weights = tf.sequence_mask(
            self.trg_size, maxlen=tf.shape(self.trg_output)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        self.cost_op = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。反向操作的实现与语言模型代码相同。
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads = tf.gradients(cost, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    def build_inference_decoder(self):
        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        max_decode_len = 100

        # 使用一个变长的TensorArray来存储生成的句子。
        init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                    dynamic_size=True, clear_after_read=False)
        # 填入第一个单词<sos>作为解码器的输入。
        init_array = init_array.write(0, SOS_ID)
        # 构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的
        # TensorArray，以及记录解码步数的一个整数step。
        init_loop_var = (self.enc_state, init_array, 0)

        # tf.while_loop的循环条件：
        # 循环直到解码器输出<eos>，或者达到最大步数为止。
        def continue_loop_condition(state, trg_ids, step):
            return tf.reduce_all(tf.logical_and(
                tf.not_equal(trg_ids.read(step), EOS_ID),
                tf.less(step, max_decode_len - 1)))

        def loop_body(state, trg_ids, step):
            # 读取最后一步输出的单词，并读取其词向量。
            trg_input = [trg_ids.read(step)]
            trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                             trg_input)
            # 这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步。
            dec_outputs, next_state = self.dec_cell.call(
                state=state, inputs=trg_emb)
            # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
            # 这一步的而输出。
            output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
            logits = (tf.matmul(output, self.softmax_weight)
                      + self.softmax_bias)
            next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
            # 将这一步输出的单词写入循环状态的trg_ids中。
            trg_ids = trg_ids.write(step + 1, next_id[0])
            return next_state, trg_ids, step + 1

        # 执行tf.while_loop，返回最终状态。
        state, trg_ids, step = tf.while_loop(
            continue_loop_condition, loop_body, init_loop_var)
        self.target_id_list = trg_ids.stack()

    def train(self, sess, src_input, src_size, trg_input, trg_output, trg_size):
        feed_dict = {
            self.src_input: src_input,
            self.src_size: src_size,
            self.trg_input: trg_input,
            self.trg_output: trg_output,
            self.trg_size: trg_size
        }
        loss, _ = sess.run([self.cost_op, self.train_op], feed_dict=feed_dict)
        return loss

    def inference(self, sess, src_input):
        feed_dict = {
            self.src_input: [src_input],
            self.src_size: [len(src_input)],
        }
        target_id_list = sess.run(self.target_id_list, feed_dict=feed_dict)
        return target_id_list

    @staticmethod
    def save(sess, saver, path, global_step=None):
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print("保存模型到{}".format(save_path))

    @staticmethod
    def restore(sess, saver, path):
        print('Model restored from {}'.format(path))
        saver.restore(sess, save_path=path)
