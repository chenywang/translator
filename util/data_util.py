# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import numpy as np
import pandas as pd
import tensorflow as tf
# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每行一句话，单词已经转化为单词编号。
from tensorflow.python.keras import preprocessing

from config import SOS_ID, MAX_LEN


def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数。
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和
# batching操作。
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据。
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds
    # 由4个张量组成：
    #   ds[0][0]是源句子
    #   ds[0][1]是源句子长度
    #   ds[1][0]是目标句子
    #   ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空（只包含<EOS>）的句子和长度过长的句子。
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(FilterLength)

    # 从图9-5可知，解码器需要两种格式的目标句子：
    #   1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
    #   2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
    # 形式并加入到Dataset中。
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据。
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))  # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


def gen_batch_train_data(train_en_path, train_zh_path, batch_size, shuffle=False):
    chunks_en = pd.read_csv(train_en_path, iterator=True, sep='\t')
    chunks_zh = pd.read_csv(train_zh_path, iterator=True, sep='\t')
    while True:
        try:
            chunk_en = chunks_en.get_chunk(batch_size)
            chunk_zh = chunks_zh.get_chunk(batch_size)
            # 合并成一个df
            chunk = chunk_en
            chunk['en'] = chunk['sentence']
            chunk['zh'] = chunk_zh['sentence']
            chunk['en'] = chunk['en'].apply(lambda line: line.split(' '))
            chunk['zh'] = chunk['zh'].apply(lambda line: line.split(' '))
            chunk['en_len'] = chunk['en'].apply(lambda line: len(line))
            chunk['zh_len'] = chunk['zh'].apply(lambda line: len(line))

            # 过滤长度不合格的
            chunk = chunk[chunk['en_len'] <= MAX_LEN]
            chunk = chunk[chunk['zh_len'] <= MAX_LEN]
            chunk = chunk[chunk['en_len'] > 1]
            chunk = chunk[chunk['zh_len'] > 1]

            # 修正target input与target output
            #   1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
            #   2.解码器的目标输出(trg_output)，形式如同"X Y Z <eos>"
            #   3.上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
            chunk['target_input'] = chunk['zh'].apply(lambda zh_list: [SOS_ID] + zh_list[:-1])
            chunk['target_output'] = chunk['zh']

            src_input, src_size, trg_input, trg_output, trg_size = np.array(chunk['en']), np.array(
                chunk['en_len']), np.array(chunk['target_input']), np.array(
                chunk['target_output']), np.array(chunk['zh_len'])

            # padding
            src_input = preprocessing.sequence.pad_sequences(src_input, maxlen=max(chunk['en_len']),
                                                             padding="post", truncating="post",
                                                             value=0)
            trg_input = preprocessing.sequence.pad_sequences(trg_input, maxlen=max(chunk['zh_len']),
                                                             padding="post", truncating="post",
                                                             value=0)
            trg_output = preprocessing.sequence.pad_sequences(trg_output, maxlen=max(chunk['zh_len']),
                                                              padding="post", truncating="post",
                                                              value=0)

            if shuffle:
                chunk.sample(frac=1.0)

            yield src_input, src_size, trg_input, trg_output, trg_size
        except StopIteration:
            break
