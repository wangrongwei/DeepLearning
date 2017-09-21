# -- coding:utf-8 --
# 对tf.multiply的学习


import tensorflow as tf
import numpy as np

X = tf.constant([[1,2],[2,2]])
Y = tf.constant([1,3])

Z = tf.multiply(X,Y)

with tf.Session() as sess:
   #sess.run(Z)
   print(sess.run(Z))








