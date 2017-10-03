# coding:utf-8
# CIFAR-10数据集图像label:
# airplane-0 automobile-1 bird-2 cat-3 deer-4 
# dog-5      frog-6     horse-7  ship-8 truck-9
import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

""" 第一部分：数据集 """
""" 定义操作数据集的类 """
class Data_set:
    pass

class Dataset():
    # __init__ 是析构函数 
    def __init__(self):
        print("init\n")
        self.data = np.zeros((50000,32,32,3))
        self.labels = np.zeros((50000,10))
        self.batch_x = np.zeros((50,32,32,3))
        self.batch_y = np.zeros((50,10))
        self.num = 0
        self.rand = 0
    """ 读取数据 """
    def read_data(self,matrix,label):
        self.data = matrix
        self.labels = label
    """ 在训练时，需要选择batch大小的数据进行 """
    """ 固定大小的batch == 50 """
    def next_batchs(self,num):
        self.num = num
        #for i in range(self.num):
        self.batch_x = self.data[self.num:self.num+50][:][:][:]
        self.batch_y = self.labels[self.num:self.num+50][:]
        #print(self.batch_y)
        return self.batch_x,self.batch_y

CIFAR10 = Data_set()
CIFAR10.manage = Dataset()

""" 第二部分：转换数据集格式 """
cifar_class = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] 
# 读取单个的batch文件
def unpickle(datafile):
    import cPickle
    with open('./cifar-10-batches-py/'+datafile,'rb') as fo:
        dict = cPickle.load(fo)
    return dict


start_time = time.time()

data1 = unpickle('data_batch_1')
data2 = unpickle('data_batch_2')
data3 = unpickle('data_batch_3')
data4 = unpickle('data_batch_4')
data5 = unpickle('data_batch_5')

# 读取5次data-batch 然后将五个数据整合到一个矩阵
X1 = data1['data']
label1 = data1['labels']
X1 = np.array(X1)
np.set_printoptions(threshold='nan')
new1 = X1.reshape(-1,3,32,32)

X2 = data2['data']
label2 = data2['labels']
X2 = np.array(X2)
np.set_printoptions(threshold='nan')
new2 = X2.reshape(-1,3,32,32)

X3 = data3['data']
label3 = data3['labels']
X3 = np.array(X3)
np.set_printoptions(threshold='nan')
new3 = X3.reshape(-1,3,32,32)

X4 = data4['data']
label4 = data4['labels']
X4 = np.array(X4)
np.set_printoptions(threshold='nan')
new4 = X4.reshape(-1,3,32,32)

X5 = data5['data']
label5 = data5['labels']
X5 = np.array(X5)
np.set_printoptions(threshold='nan')
new5 = X5.reshape(-1,3,32,32)

# label里边成员还不是整型，为字符型
label = np.vstack((label1,label2,label3,label4,label5))
X = np.vstack((new1,new2,new3,new4,new5))
end_time = time.time() - start_time


print("end_time:{0:f}\n".format(end_time))
# 因为使用imshow将一个矩阵显示为RGB图片，需要
# 将三个32*32的矩阵合成一个32*32*3的三维矩阵
cifar10_label_1 = label.reshape(-1,1)
cifar10_label = np.zeros((50000,10))

for i in range(50000):
    for j in range(10):
        if cifar10_label_1[i] == j:
            cifar10_label[i][j] = 1
        else:
            cifar10_label[i][j] = 0


cifar10 = X.transpose((0,2,3,1))

CIFAR10.manage.read_data(cifar10,cifar10_label)
# imshow显示的图片格式应该是
# (n,m) or (n,m,3) or (n,m,4)
# 显示最后得到的rgb图片

#print(label.shape)
#plt.imshow(cifar10[49])
#plt.title(cifar_class[int(cifar10_label[49])])

#plt.legend()
#plt.show()

""" 第三部分：开始布置卷积神经网络 """
def kernel_variable(shape):
    kernel = tf.truncated_normal(shape=shape,stddev=5e-2)
    return tf.Variable(kernel)
def bias(shape):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder('float',shape=[None,32,32,3])
y_ = tf.placeholder('float',shape=[None,10])

# 卷积神经网络第一层参数
W1 = kernel_variable([5,5,3,64])
b1 = bias([64])
#x_image = tf.reshape(x,[-1,32,32,1])
y1 = conv2d(x,W1) + b1
y1_pool = tf.nn.relu(max_pool2x2(y1))

# 卷积神经网络第二层
W2 = kernel_variable([5,5,64,64]) 
b2 = bias([64])
y2 = conv2d(y1_pool,W2) + b2
y2_pool = tf.nn.relu(max_pool2x2(y2))

# 开始使用优化方法学习
Weight_1 = kernel_variable([8*8*64,1024])
bias_1   = bias([1024])
y2_pool_new = tf.reshape(y2_pool,[-1,8*8*64])
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

for i in range(1000):
    batch_xs,batch_ys = CIFAR10.manage.next_batchs(i*50)
    #print(batch_xs.shape)
    #print('\n')
    #print(batch_ys.shape)
    if i%50 ==0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys,
            keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
    
# print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,
#   y_:mnist.test.labels,keep_prob:1.0}))



