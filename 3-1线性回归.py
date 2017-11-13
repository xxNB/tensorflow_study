# -*- coding: utf-8 -*-
"""
Created on 2017/11/13 下午4:37
@author: SimbaZhang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# print(x_data)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义俩个placehold, float32:浮点型
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义一个神经网络的中间层
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
biase_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biase_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biase_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biase_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)


#定义一个二次代价函数, reduce_mean:求平均值
loss = tf.reduce_mean(tf.square(prediction - y))
#定义一个梯度下降法来训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train, feed_dict={x:x_data, y:y_data})

    #获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
#使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# print(x_data)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义俩个placehold, float32:浮点型
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义一个神经网络的中间层
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
biase_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biase_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biase_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biase_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)


#定义一个二次代价函数, reduce_mean:求平均值
loss = tf.reduce_mean(tf.square(prediction - y))
#定义一个梯度下降法来训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train, feed_dict={x:x_data, y:y_data})

    #获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

