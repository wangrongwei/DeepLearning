# -- coding:utf-8 --
# 将图片从x=1000往下切，只显示往下的[3000,-1,-1]部分:代表最后

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8",[None,None,3])
slice = tf.slice(image,[0,0,0],[3000,-1,-1])

with tf.Session() as session:
    result = session.run(slice,feed_dict={image:raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()







