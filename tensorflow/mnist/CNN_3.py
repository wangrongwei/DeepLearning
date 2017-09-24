# --coding:utf-8 --
# 描述：三层的卷积神经网络

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import input_data
mnist = input_data.read_data_sets("~/work/learn_data/MNIST_data",one_hot='True')

sess = tf.InteractiveSession()


# 函数功能描述：将图片形式的矩阵（32x32x3）转化成（3x32x32）
def rgb_matrix(row,col,channels,rgb):
    pic = np.array(rgb)
    print(pic.shape)
    matrix = np.zeros((channels,row,col))
    
    for k in range(channels):
        for i in range(row):
            for j in range(col):
                matrix[k][i][j] = pic[i][j][k]
    return matrix
def show_matrix(row,col,channels,matrix):
    for j in range(channels):
        plt.subplot(row,col,j+1)
        plt.imshow(matrix[j],cmap='gray') 
        plt.axis('off')
    plt.legend()
    plt.show()

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
y_ = tf.placeholder('float',shape=[None,10])

# 卷积神经网络第一层参数
W1 = kernel_variable([5,5,1,32])
b1 = bias([32])
x_image = tf.reshape(x,[-1,28,28,1])
y1 = conv2d(x_image,W1) + b1
y1_pool = tf.nn.relu(max_pool2x2(y1))

# 卷积神经网络第二层
W2 = kernel_variable([5,5,32,64]) 
b2 = bias([64])
y2 = conv2d(y1_pool,W2) + b2
y2_pool = tf.nn.relu(max_pool2x2(y2))

# 卷积神经网络第三层
W3 = kernel_variable([3,3,64,128])
b3 = bias([128])
y3 = conv2d(y2_pool,W3) + b3
y3_pool = tf.nn.relu(tf.nn.max_pool(y3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))
print("y3_pool")
print(y3_pool.shape)

# 开始使用优化方法学习
Weight_1 = kernel_variable([4*4*128,1024])
bias_1   = bias([1024])
y4 = tf.reshape(y3_pool,[-1,4*4*128])
y4_result = tf.nn.relu(tf.matmul(y4,Weight_1) + bias_1)

Weight_2 = kernel_variable([1024,10])
bias_2   = bias([10])
keep_prob = tf.placeholder('float')

y5 = tf.nn.dropout(y4_result,keep_prob)
y5_result = tf.nn.softmax(tf.matmul(y5,Weight_2) + bias_2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y5_result))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(y5_result,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'))


init = tf.global_variables_initializer()

sess.run(init)

for i in range(4000):
    batch = mnist.train.next_batch(50)
    if i%50 ==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))




