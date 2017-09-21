# --coding:utf-8 --
import tensorflow as tf
import numpy as np

# 可视化tensorboard需要
import matplotlib.pyplot as plt
# 设置学习率
learning_rate = 0.02 


# 对于测试的集合相当于和这个训练的集合是一个大类，然后有些取出来做训练，
# 有些做测试用
X = np.asarray([1.,2.,3.,4.,5.,1.1,6.,7.],dtype="float")
Y = np.asarray([2.,3.,5.,9.,11.,1.5,12.,19.],dtype="float")

train_X = tf.placeholder('float')
train_Y = tf.placeholder('float')


W = tf.Variable(np.random.randn(),name="weight")
b = tf.Variable(np.random.randn(),name="bias")

Y_ = tf.add(tf.multiply(W,train_X),b)
cost = tf.reduce_sum(tf.pow(Y_-train_Y,2))/(2*8)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 为什么sess.run()放在这里才伴随循环一直在更新Variable
    # init只运行一次
    sess.run(init)
    for i in range(1000):
        for (x,y) in zip(X,Y):
            sess.run(optimizer,feed_dict={train_X:x,train_Y:y})
        if i%100==0:
            c = sess.run(cost,feed_dict={train_X:X,train_Y:Y})
            print("cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print("Optimize finished !!")
    train_cost = sess.run(cost,feed_dict={train_X:X,train_Y:Y})
    print("train_cost="+str(train_cost))

    # 图像显示
    plt.plot(X, Y, 'ro', label='Original data')
    plt.plot(X, sess.run(W) * X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    # 可视化整个过程
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./tensorboard/basic",sess.graph)
    #print(sess.run(optimizer,{train_X:X,train_Y:Y}))


