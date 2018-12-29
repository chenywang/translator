# -*- coding: utf-8 -*-
# @Author: disheng
import os

PROJECT_PATH = os.path.abspath(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.pardir))

project_path = PROJECT_PATH
data_path = PROJECT_PATH + '/data/'
log_path = project_path + '/logs/'

# 假设输入数据已经用9.2.1小节中的方法转换成了单词编号的格式。
SRC_TRAIN_DATA = data_path + "train.en"  # 源语言输入文件。
TRG_TRAIN_DATA = data_path + "train.zh"  # 目标语言输入文件。

# 词汇表文件
SRC_VOCAB = data_path + "en.vocab"
TRG_VOCAB = data_path + "zh.vocab"

HIDDEN_SIZE = 1024  # LSTM的隐藏层规模。
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数。
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小。
BATCH_SIZE = 100  # 训练数据batch的大小。
NUM_EPOCH = 5  # 使用训练数据的轮数。
KEEP_PROB = 0.8  # 节点不被dropout的概率。
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50  # 限定句子的最大单词数量。
SOS_ID = 1  # 目标语言词汇表中<sos>的ID。
EOS_ID = 2
