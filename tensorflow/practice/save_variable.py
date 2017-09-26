# -- coding:utf-8 --

import tensorflow as tf

import numpy as np

data_sets = input_data.read_data_sets(FLAGS.train_dir,FLAGS.fake_data)
images_placeholder = tf.placeholder(tf.float32,shape=(batch_size,mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32,shape=(batch_size))




v1 = tf.Variable(tf.random_normal([784,200],stddev=0.35))

init = tf.global_variables_initializer()
#saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_v1 = tf.train.Saver({"my_v1":v1})




