# --coding:utf-8 --
import tensorflow as tf
import numpy as np

import input_data
mnist = input_data.read_data_sets("~/work/learn_data/MNIST_data/",one_hot=True)

import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def show_n_pic(n,channels,matrix):
    for i in range(n):
        for j in range(channels):
            plt.subplot(n,channels,channels*i+j+0)
            plt.imshow(matrix[i][0:][0:][j],cmap='gray')
            plt.axis('off')
    plt.legend()
    plt.show()


def weight_variable(shape):
    weight = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weight)
def bias(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def maxpool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder('float',shape=[None,784])

W = weight_variable([5,5,1,32])
b = bias([32])
x_image = tf.reshape(x,[-1,28,28,1])
y = tf.nn.relu(conv2d(x_image,W)+b)

init = tf.global_variables_initializer()

sess.run(init)
    
batch_xs,batch_ys = mnist.train.next_batch(10)
sess.run(y,feed_dict={x:batch_xs})
Y = sess.run(y,feed_dict={x:batch_xs})
Y = Y*255
print(Y.shape)
show_n_pic(10,28,Y)




