# -- coding:utf-8 --
# 实现两个矩阵相乘

import tensorflow as tf

b = tf.Variable(tf.zero([100]))
# 生成784*100的随机矩阵
W = tf.Variable(tf.random_uniform([784,100],-1,1))

x = tf.placeholder(name="x")
relu = tf.nn.relu(tf.matmul(W,x)+b) #ReLU(Wx+b)

C = [...]
s = tf.Session()

for step in range(0,10):
    input=...construct 100-D input array ...
    result = s.run(C,feed_dict={x:input})
    print(step,result)
    












