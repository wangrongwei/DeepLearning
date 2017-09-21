# --coding:utf-8 --
import math

import tensorflow as tf


sqrt = math.sqrt
points = tf.constant([[8,0],
                      [4,-sqrt(24)],
                      [-sqrt(56),2],
                      [-sqrt(46),3],
                      [sqrt(14),5]],dtype="float64")

A = tf.constant([[64,0,0,8,0],
                 [16,24,sqrt(384),4,-sqrt(24)],
                 [56,4,-sqrt(224),-sqrt(56),2],
                 [46,9,-sqrt(414),-sqrt(46),3],
                 [14,25,sqrt(350),sqrt(14),5]],dtype="float64")

B = tf.constant([[-1],[-1],[-1],[-1],[-1]],dtype="float64")

X = tf.matrix_solve(A,B)
with tf.Session() as sess:
    result = sess.run(X)
    A1,B1,C1,D1,E1 = result.flatten()
    print("E:{A1}x**2+{B1}y**2+{C1}xy+{D1}x+{E1}y".format(**locals()))








