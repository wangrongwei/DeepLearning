# --coding:utf-8--
# 使用cifar-10数据集搭建一层卷积神经网络


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def unpickle(file):
    import cPickle
    with open('./cifar-10-batches-py/'+file,'rb') as fo:
        dict = cPickle.load(fo)
    return dict





