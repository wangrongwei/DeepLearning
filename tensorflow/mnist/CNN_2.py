# --coding:utf-8 --
# 描述：展示CNN每一层每个图片的变化

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

# 开始使用优化方法学习
Weight_1 = kernel_variable([7*7*64,1024])
bias_1   = bias([1024])
y2_pool_new = tf.reshape(y2_pool,[-1,7*7*64])
y2_pool_result = tf.nn.relu(tf.matmul(y2_pool_new,Weight_1) + bias_1)
keep_prob = tf.placeholder('float')
y2_pool_drop = tf.nn.dropout(y2_pool_result,keep_prob)

Weight_2 = kernel_variable([1024,10])
bias_2   = bias([10])
result = tf.nn.softmax(tf.matmul(y2_pool_drop,Weight_2) + bias_2)

cross_entropy = -tf.reduce_sum(y_*tf.log(result))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(result,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'))


init = tf.global_variables_initializer()

sess.run(init)

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i%50 ==0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


# 开始显示输入层图片
#x1_src = tf.reshape(x,[-1,28,28])
#x1_show = sess.run(x1_src,feed_dict={x:batch_xs})
#plt.imshow(x1_show[0],cmap="gray")
#plt.legend()
#plt.show()

# 显示第二层网络
#y2_show = sess.run(y2_pool,feed_dict={x:batch_xs})
#y2_tran = rgb_matrix(7,7,64,y2_show[0])
#y2_show = tf.reshape(y2_show,[-1,28,28])
#show_matrix(8,8,64,y2_tran)


#result = sess.run(batch_image,feed_dict={x:batch_xs})
#print(result.shape)

