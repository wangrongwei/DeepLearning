# -- coding:utf-8 --
# mnist学习
# 需要记住的是每一个图片本身就是二值图像，使用1*784表示即可
import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

batch_xs, batch_ys = mnist.train.next_batch(1)

pic = batch_xs.reshape(28,28)
pic = pic*255
print(batch_xs)

plt.imshow(pic,cmap='gray')

plt.legend()
plt.show()




