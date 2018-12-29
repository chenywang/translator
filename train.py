# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import os
import time

import tensorflow as tf
from tensorflow.python.platform import flags

# 使用给定的模型model上训练一个epoch，并返回全局步数。
# 每训练200步便保存一个checkpoint。
from config import model_path, TRG_TRAIN_DATA, SRC_TRAIN_DATA, model_name
from model.seq2seq_model import NMTModel
from util.data_util import gen_batch_train_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('epoch', 100, 'Maximum # of training epochs')
flags.DEFINE_boolean('retrain', True, 'retrain the model')
FLAGS = flags.FLAGS


def load_or_create_model(sess, model, saver):
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and not FLAGS.retrain:
        print('Reloading model parameters...')
        model.restore(sess, saver, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print('Created new model parameters...')
        sess.run(tf.global_variables_initializer())


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", initializer=initializer):
        model = NMTModel('train')
    config_proto = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    saver = tf.train.Saver()
    with tf.Session(config=config_proto) as sess:
        load_or_create_model(sess, model, saver)
        for epoch_idx in range(FLAGS.epoch):

            iterator = gen_batch_train_data(SRC_TRAIN_DATA, TRG_TRAIN_DATA, FLAGS.batch_size, shuffle=False)
            for batch_index, (src_input, src_size, trg_input, trg_output, trg_size) in enumerate(iterator):
                start_time = time.time()

                # 训练
                step_loss = model.train(sess, src_input, src_size, trg_input, trg_output, trg_size)

                # 展示
                time_elapsed = time.time() - start_time
                print("第{epoch}个epoch的第{batch}个batch："
                      "loss为{loss}，"
                      "花费时间为{time_elapsed}，"
                      "batch_size为{batch_size}"
                      .format(epoch=epoch_idx,
                              batch=batch_index,
                              loss=step_loss,
                              time_elapsed=time_elapsed,
                              batch_size=src_input.shape[0]))

                # 记录情况
                # log_writer.add_summary(summary, model.global_step.eval())

                if batch_index % 100 == 0:
                    print('已完成{}epoch，保存该模型中'.format(epoch_idx))
                    checkpoint_path = os.path.join(model_path, model_name)
                    model.save(sess, saver, checkpoint_path, global_step=epoch_idx)


if __name__ == "__main__":
    main()
