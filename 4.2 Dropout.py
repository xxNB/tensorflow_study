# -*- coding: utf-8 -*-
"""
Created on 2017/11/13 下午4:39
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

#
keep_prob = tf.placeholder(tf.float32)


# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([784,2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000,2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000,1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000,10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)


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
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))# arg_max 返回最大的值所在位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})

        print("Iter" + str(epoch) + ", Testing_Accuracy " + str(test_acc) + ", Train_Accuracy "+  str(train_acc))