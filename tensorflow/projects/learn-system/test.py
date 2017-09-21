# --coding:utf-8 --
# CIFAR-10数据集图像label:
# airplane-0 automobile-1 bird-2 cat-3 deer-4 
# dog-5      frog-6     horse-7  ship-8 truck-9

import tensorflow as tf
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt


np.set_printoptions(threshold='nan')

pic1 = plt.imread("../lenna.jpg")
pic = np.array(pic1)

#print(pic)
red = np.zeros((32,32))
plt.imshow(red)

for i in range(32):
    for j in range(32):
        red[i][j] = pic[i][j][2]
print(red)
plt.imshow(pic)

plt.legend()
plt.show()




