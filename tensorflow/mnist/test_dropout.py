import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    matrix_4x4 = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],dtype='float')
    matrix = sess.run(matrix_4x4)
    print(matrix)
    
    dropout_matrix = sess.run(tf.nn.dropout(matrix,0.5)) 
    
    print(dropout_matrix)










