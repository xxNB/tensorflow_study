# -*- coding: utf-8 -*-
"""
Created on 2017/11/13 下午4:40
@author: SimbaZhang
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('sttdev'):
            sttdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('sttdev', sttdev)
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 定义命名空间
with tf.name_scope('input'):
    # 定义俩个Placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

# 定义命名空间

with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]), name='w')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 创建一个简单的神经网络
# prediction = tf.nn.softmax(tf.matmul(x, W)+b)

# 定义一个二次代价函数, reduce_mean:求平均值
# loss = tf.reduce_mean(tf.square(prediction - y))

# 交叉熵替代二次代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
# 定义一个梯度下降法来训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
with tf.name_scope('train'):
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# 结果存放在一个bool型列表中
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # arg_max 返回最大的值所在位置
    # 求准确率
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ", Testing_Accuracy " + str(acc))