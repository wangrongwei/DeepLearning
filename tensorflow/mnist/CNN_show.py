# --coding:utf-8 --
# 描述：展示CNN每一层每个图片的变化

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import input_data
mnist = input_data.read_data_sets("~/work/learn_data/MNIST_data",one_hot='True')

sess = tf.InteractiveSession()

def kernel_variable(shape):
    kernel = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(kernel)
def bias(shape):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,784])


W1 = kernel_variable([5,5,1,32])
b1 = bias([32])
x_image = tf.reshape(x,[-1,28,28,1])
y1 = conv2d(x_image,W1) + b1
y1_pool = tf.nn.relu(max_pool2x2(y1))

W2 = kernel_variable([5,5,32,64]) 
b2 = bias([64])
y2 = conv2d(y1_pool,W2) + b2
y2_pool = tf.nn.relu(max_pool2x2(y2))

batch_image = tf.reshape(y2_pool,[-1,7*7*64])
init = tf.global_variables_initializer()

batch_xs,batch_ys = mnist.train.next_batch(10)
sess.run(init)

result = sess.run(batch_image,feed_dict={x:batch_xs})
print(result.shape)








