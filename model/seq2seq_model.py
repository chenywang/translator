# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from config import HIDDEN_SIZE, NUM_LAYERS, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, SHARE_EMB_AND_SOFTMAX, KEEP_PROB, \
    MAX_GRAD_NORM, SOS_ID, EOS_ID


class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    # 在forward函数中定义模型的前向计算图。
    # src_input, src_size, trg_input, trg_label, trg_size分别是上面
    # MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量。
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout。
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 使用dynamic_rnn构造编码器。
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类
        # 张量的tuple，每个LSTMStateTuple对应编码器中的一层。
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size,
        # max_time, HIDDEN_SIZE]。Seq2Seq模型中不需要用到enc_outputs，而
        # 后面介绍的attention模型会用到它。
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # 使用dyanmic_rnn构造解码器。
        # 解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步
        # 顶层LSTM的输出。dec_outputs的维度是 [batch_size, max_time,
        # HIDDEN_SIZE]。
        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        with tf.variable_scope("decoder"):
            dec_outputs, _ = dynamic_rnn(
                self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

        # 计算解码器每一步的log perplexity。这一步与语言模型代码相同。
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰
        # 模型的训练。
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。反向操作的实现与语言模型代码相同。
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))
        return cost_per_token, train_op

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        # 将输入句子整理为大小为1的batch。
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器。这一步与训练时相同。
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN = 100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, SOS_ID)
            # 构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的
            # TensorArray，以及记录解码步数的一个整数step。
            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN - 1)))

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
            return trg_ids.stack()
