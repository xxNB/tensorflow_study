# -*- coding: utf-8 -*-
"""
Created on 2017/11/13 下午4:36
@author: SimbaZhang
"""

import tensorflow as tf
import numpy as np

#使用numpy生产100各随机点
x_data = np.random.rand(100)
y_data = x_data*0.5 + 0.3

#构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

#定义一个二次代价函数, reduce_mean:求平均值
loss = tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)

# 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i % 20 == 0:
            print(i, sess.run([k ,b]))