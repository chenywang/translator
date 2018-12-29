# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import codecs
import os

import tensorflow as tf

from config import SRC_VOCAB, model_path, TRG_VOCAB
from model.seq2seq_model import NMTModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # 根据中文词汇表，将翻译结果转换为中文文字。
    print("获得中文词汇表")
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]

    # 根据英文词汇表，将测试句子转为单词ID。
    print("获得英文词汇表")
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))

    sess = tf.Session()
    # 定义训练用的循环神经网络模型。
    print("重建模型中")
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel('predict')
    saver = tf.train.Saver()
    latest_cpt_file = tf.train.latest_checkpoint(model_path)
    model.restore(sess, saver, latest_cpt_file)
    while True:
        input_english_text = "This is a test . <eos>"
        # input_english_text = input('输入关键词:\n').strip()
        english_id_list = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                           for token in input_english_text.split()]

        output_ids = model.inference(sess, english_id_list)

        output_text = ''.join([trg_vocab[x] for x in output_ids])

        # 输出翻译结果。
        print(output_text)


if __name__ == "__main__":
    main()
