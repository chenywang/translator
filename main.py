# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import codecs
import os

import pandas as pd
import tensorflow as tf

from config import SRC_VOCAB, model_path, TRG_VOCAB
from model.seq2seq_model import NMTModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # 根据中文词汇表，将翻译结果转换为中文文字。
    print("获得中英文词汇表")
    # src_df = pd.read_csv(SRC_VOCAB, sep='\t')
    # src_vocab = list(src_df['word'])
    # src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))

    # trg_df = pd.read_csv(TRG_VOCAB, sep='\t')
    # trg_vocab = list(trg_df['word'])
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]

    print("重建模型中")
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel('predict')
    saver = tf.train.Saver()
    latest_cpt_file = tf.train.latest_checkpoint(model_path)
    sess = tf.Session()
    model.restore(sess, saver, latest_cpt_file)
    while True:
        # input_english_text = "This is a test . <eos>"
        input_english_text = input('输入关键词:\n').strip()
        input_english_text += ' . <eos>'
        english_id_list = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                           for token in input_english_text.split()]
        print(english_id_list, len(english_id_list))
        output_ids = model.inference(sess, english_id_list)

        output_text = ''.join([trg_vocab[x] for x in output_ids])
        print(output_ids, len(output_ids))

        # 输出翻译结果。
        print(output_text)
        # break


if __name__ == "__main__":
    main()
