import tensorflow as tf

points = tf.constant([[2,1],[0,5],[-1,2]],dtype='float64')

A = tf.constant([[2,1,1],[0,5,1],[-1,2,1]],dtype='float64')

B = -tf.constant([[5],[25],[5]],dtype='float64')

X = tf.matrix_solve(A,B)

with tf.Session() as sess:
    result = sess.run(X)
    D,E,F = result.flatten()
    print("E:x**2 + y**2 + {D}x + {E}y + {F} = 0".format(**locals()))





