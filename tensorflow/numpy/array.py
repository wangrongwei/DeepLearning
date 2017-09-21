# --coding:utf-8 --
import numpy as np
from numpy import pi

x = np.array([[1,0,0],[2,3,1]])

print(x)
print(x.shape)
print(x.data)

# 表示每一个元素占的字节数
print(x.itemsize)

print(x.dtype)

y = np.array([1,3,6],dtype=complex)
print(y)
print('\n')
zeros_matrix = np.zeros((2,5))

print(zeros_matrix)

ones_matrix = np.ones((3,4))
print(ones_matrix)

empty1 = np.empty((2,3))
print(empty1)

z = np.arange(0,10,8)
print(z)
#h = np.linspace(0,2*pi,100)
#print(np.sin(h))

A = np.array([[1,1],[0,1]])
B = A
B.reshape(1,-1)
print(A)
print(B)

