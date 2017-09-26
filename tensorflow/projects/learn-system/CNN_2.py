# coding:utf-8
# CIFAR-10数据集图像label:
# airplane-0 automobile-1 bird-2 cat-3 deer-4 
# dog-5      frog-6     horse-7  ship-8 truck-9
import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt

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

print(label5[9991])
label = np.vstack((label1,label2,label3,label4,label5))
X = np.vstack((new1,new2,new3,new4,new5))
end_time = time.time() - start_time


print("end_time:{0:f}\n".format(end_time))
# 因为使用imshow将一个矩阵显示为RGB图片，需要
# 将三个32*32的矩阵合成一个32*32*3的三维矩阵

#def next_batch(matrix,batchs):
    

# 下面就是先将这三个矩阵（32*32）转化为1024*1的向量
# 然后使用hstack的功能将每个矩阵上相同位置的值合成
# 一个RGB像素点--->[r,g,b]
# 最后得到 1024*3的矩阵
red   = X[49991][0].reshape(1024,1)
green = X[49991][1].reshape(1024,1)
blue  = X[49991][2].reshape(1024,1)

pic = np.hstack((red,green,blue))

# 打印最开始的32*32的矩阵，
# 因为为RGB图像，所以为有三个32*32的矩阵

# 重新设置pic的形状
pic_rgb = pic.reshape(32,32,3)
# imshow显示的图片格式应该是
# (n,m) or (n,m,3) or (n,m,4)
# 显示最后得到的rgb图片
plt.imshow(pic_rgb)

plt.legend()
plt.show()



