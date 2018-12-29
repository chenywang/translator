# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import tensorflow as tf

# 使用给定的模型model上训练一个epoch，并返回全局步数。
# 每训练200步便保存一个checkpoint。
from config import log_path, BATCH_SIZE, NUM_EPOCH, TRG_TRAIN_DATA, SRC_TRAIN_DATA
from model.seq2seq_model import NMTModel
from util.data_util import MakeSrcTrgDataset


def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch。
    # 重复训练步骤直至遍历完Dataset中所有数据。
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供。
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("第 %d batch, 平均交叉熵 %.3f" % (step, cost))
            # 每200步保存一个checkpoint。
            if step % 200 == 0:
                saver.save(session, log_path, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None,
                           initializer=initializer):
        train_model = NMTModel()

    # 定义输入数据。
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    # 训练模型。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In epoch: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == "__main__":
    main()
