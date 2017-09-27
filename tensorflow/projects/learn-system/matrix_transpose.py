# --coding:utf-8 --
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# 读取单个的batch文件
def unpickle(file):
    import cPickle
    with open('./cifar-10-batches-py/'+file,'rb') as fo:
        dict = cPickle.load(fo)
    return dict


mydata = unpickle('data_batch_1')
X = mydata['data']
label = mydata['labels']
X = np.array(X)
np.set_printoptions(threshold='nan')

new = X.reshape(10000,3,32,32)


# 因为使用imshow将一个矩阵显示为RGB图片，需要
# 将三个32*32的矩阵合成一个32*32*3的三维矩阵

# 方法一，单个图像转化，可理解、可计算,可以作为方法二的验证 

# 下面就是先将这三个矩阵（32*32）转化为1024*1的向量
# 然后使用hstack的功能将每个矩阵上相同位置的值合成
# 一个RGB像素点--->[r,g,b]
# 最后得到 1024*3的矩阵
red   = new[1990][0].reshape(1024,1)
green = new[1990][1].reshape(1024,1)
blue  = new[1990][2].reshape(1024,1)

pic = np.hstack((red,green,blue))
# 重新设置pic的形状
pic_rgb = pic.reshape(32,32,3)

# 方法二 使用np.transpose,用法如下，可以使用方法一进行验证两种方法是否一样

# 这里的transpose函数里边的参数代表对于
# [batch][3][32][32] 对于维度编号是0,1,2,3
# 如果转化为[batch][32][32][3]，其中0位置不变，把1插到原来0、2之间
pic_test = new.transpose((0,2,3,1))
# 重新设置pic的形状
pic_rgb = pic.reshape(32,32,3)
# imshow显示的图片格式应该是
# (n,m) or (n,m,3) or (n,m,4)
# 显示最后得到的rgb图片
plt.imshow(pic_test[1990])

plt.legend()
plt.show()


