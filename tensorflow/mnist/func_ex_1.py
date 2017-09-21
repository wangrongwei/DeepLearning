# -- coding:utf-8 --
# 实现方程：y = 5x^2-3x+15

import tensorflow as tf
import numpy as np

x = tf.Variable(0,name='x')

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        data = np.random.randint(1000)
        print(data)
        x = x + (data-x)/(i+1)
        print(sess.run(x))






