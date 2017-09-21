# -- coding:utf-8 --
# mnist学习
# 需要记住的是每一个图片本身就是二值图像，使用1*784表示即可
#

import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf
import numpy as np

learn_rate = 0.008

W = tf.Variable(tf.zeros([784,10]),dtype='float')
b = tf.Variable(tf.zeros([1,10]),dtype='float')

x = tf.placeholder('float',[None,784])

y_ = tf.placeholder('float',[None,10])

y = tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_test = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_test,feed_dict={x:batch_xs,y_:batch_ys})
    
    # tf.equal返回的是True或者False
    # 因此在下面使用tf.cast转化成float
    prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    # tf.reduce_mean() 是求平均数
    accuracy = tf.reduce_mean(tf.cast(prediction,'float'))
    print('print last accuracy')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))








