
# coding:utf-8

# 建立一个空的结构图
class Data_set:
    pass

class Dataset():
    # __init__ 是析构函数 
    def __init__(self):
        print("init\n")
        self.data = []
        self.labels = []
        self.batchs = []
        self.num = 1
    """ 读取数据 """
    def read_data(self,matrix,label):
        self.data = matrix
        self.labels = label
    """ 在训练时，需要选择batch大小的数据进行 """
    def next_batchs(self,num):
        self.num = num
        self.batch_x = data[0:num-1]
        self.batch_y = labels[0:num-1] 
        return batch_x,batch_y




