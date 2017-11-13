# -*- coding: utf-8 -*-
"""
Created on 2017/11/13 下午4:38
@author: SimbaZhang
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义俩个Placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

#定义一个二次代价函数, reduce_mean:求平均值
# loss = tf.reduce_mean(tf.square(prediction - y))

#交叉熵替代二次代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#定义一个梯度下降法来训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# 结果存放在一个bool型列表中
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))# arg_max 返回最大的值所在位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter" + str(epoch) + ", Testing_Accuracy " + str(acc))