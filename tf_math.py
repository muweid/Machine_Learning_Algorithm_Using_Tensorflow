#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class tf_Math:
    def __init__(self):
        pass

    def tf_square(self, n):
        """
        Not an effective method at all.
        Just want to demonstrate the capability of Tensorflow
        May return -1 (error) for large numbers. Tested till 1E5.
        Again, very slow...
        :param n: a single value
        :return: the square root of value n
        """
        epsilon = 1e-3
        max_step = 100000

        x = tf.placeholder(tf.float32, [1])
        ans = tf.Variable(1., tf.float32)
        loss = tf.square((tf.square(ans) - x))
        train_op = tf.train.MomentumOptimizer(1e-6, momentum=0.9).minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for step in range(max_step):
                sess.run(train_op, feed_dict={x: [n]})
                if step % 100 == 0:
                    temp = sess.run(ans)
                    if (temp * temp - n) ** 2 < epsilon:
                        return (sess.run(ans))

            return -1
