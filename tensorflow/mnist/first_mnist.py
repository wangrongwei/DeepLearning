# -- coding:utf-8 --
# mnist学习
# 需要记住的是每一个图片本身就是二值图像，使用1*784表示即可
import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf
import numpy as np

learn_rate = 0.01

x = tf.placeholder("float",[None,784])

# 为什么此处为784*10，10的含义就是我们对于每一张图片
# 可能是0-9，需要我们每一个数字需要不同的权重
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# y 代表训练到的值,但是这个y不是0~9，而是一个张量：1x10
y = tf.nn.softmax(tf.matmul(x,w) + b)
# y_代表当前这个图片的真实值
# 相当于需要测试的数据分布
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()

# 训练次数
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    npw = np.array(sess.run(w))
    np.set_printoptions(threshold='nan')
    print('W=',npw,'b',sess.run(b))
    
    # 这里的1表示方向：表示什么方向的最大值 
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    print("print correct_prediction shape")
    print(correct_prediction.shape)
    # 求平均值 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(mnist.test.labels.shape)


