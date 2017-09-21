# --coding:utf-8 --
# 根据两点计算方程
#  {-a/b,1/b}*|x1 x2|  = {1,1}
#             |y1 y2|
import tensorflow as tf

x1 = tf.constant(2,dtype=tf.float32)
y1 = tf.constant(9,dtype=tf.float32)
point1 = tf.stack([x1,y1])

x2 = tf.constant(-1,dtype=tf.float32)
y2 = tf.constant(3,dtype=tf.float32)
point2 = tf.stack([x2,y2])

X = tf.transpose(tf.stack([point1,point2]))

B = tf.ones((1,2),dtype=tf.float32)
parameters = tf.matmul(B,tf.matrix_inverse(X))

with tf.Session() as session:
    A = session.run(parameters)

b = 1/A[0][1]
a = -b*A[0][0]
print("Equation:y = {a}x + {b}".format(a=a,b=b))






