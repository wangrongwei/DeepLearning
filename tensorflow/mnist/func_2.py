# -- coding:utf-8 --
# 使用tensorflow实现一个简单的函数
# 使用placeholder实现函数 y = (node1+node2)*x
import tensorflow as tf

# const1 = tf.constant(3,"float32")
# const2 = tf.constant(2,"float32")
node1 = tf.placeholder("float32")
node2 = tf.placeholder("float32")

node = node1 + node2
x = tf.placeholder("float32")


y = node * x
sess = tf.Session()

print(sess.run(y,{node1:1,node2:2,x:2}))



