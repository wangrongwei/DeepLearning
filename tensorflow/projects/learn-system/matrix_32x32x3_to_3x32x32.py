# --coding:utf-8 --
# 将读取到的图片转化成三个矩阵
import tensorflow as tf
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

# 函数功能描述：将图片形式的矩阵（32x32x3）转化成（3x32x32）
def rgb_matrix(rgb):
    pic = np.array(rgb)
    red = np.zeros((32,32))
    green = np.zeros((32,32))
    blue = np.zeros((32,32))
    
    for i in range(32):
        for j in range(32):
            red[i][j] = pic[i][j][0]
            green[i][j] = pic[i][j][1]
            blue[i][j] = pic[i][j][2]
    return red,green,blue


np.set_printoptions(threshold='nan')

pic1 = plt.imread("../lenna.jpg")

# 调用函数得到三个分开的图片：r,g,b
test1,test2,test3 = rgb_matrix(pic1)

# 显示上面的之一图片
print(test2)
plt.imshow(test2)

plt.legend()
plt.show()


