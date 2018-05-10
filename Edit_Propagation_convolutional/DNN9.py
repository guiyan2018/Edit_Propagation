from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
from math import ceil

import numpy as np
import tensorflow as tf


sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def atrous_conv2d(x, W, dilation):
    return tf.nn.atrous_conv2d(x, W, dilation, padding='VALID')


def max_pool_1x1(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')




class DNN(object):

    def __init__(self, channel):
        self.batchsize = 20
        self.channel = channel

        self.W_conv1 = weight_variable([3, 3, (self.channel+3), 64])
        self.b_conv1 = bias_variable([64])

        self.W_conv2 = weight_variable([3, 3, 64, 128])
        self.b_conv2 = bias_variable([128])

        self.W_conv3 = weight_variable([3, 3, 128, 256])
        self.b_conv3 = bias_variable([256])

        self.W_fcn1 = weight_variable([1, 1, 256, 256])
        self.b_fcn1 = bias_variable([256])

        self.W_fcn2 = weight_variable([1, 1, 256, self.channel])
        self.b_fcn2 = bias_variable([self.channel])

    # 2*2 pool，atrous rate为2，感受野11。c-c-p-a-f-s
    def net(self, X):
        x_image = tf.reshape(
            X, [-1, self.h, self.w, (self.channel+3)])
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)
        h_pool1 = max_pool_1x1(h_conv2)
        atrous_conv2d1 = tf.nn.relu(atrous_conv2d(h_pool1, self.W_conv3, 2) + self.b_conv3)

        y_fcn1 = tf.nn.relu(conv2d(atrous_conv2d1, self.W_fcn1) + self.b_fcn1)
        y_fcn2 = conv2d(y_fcn1, self.W_fcn2) + self.b_fcn2
        return y_fcn2

    def prenet(self, X):
        x_image = tf.reshape(
            X, [-1, self.h, self.w, (self.channel+3)])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='VALID') + self.b_conv1)
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, self.W_conv2, strides=[1, 1, 1, 1], padding='VALID') + self.b_conv2)
        h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
        atrous_conv2d1 = tf.nn.relu(tf.nn.atrous_conv2d(h_pool1, self.W_conv3, 2, padding='VALID') + self.b_conv3)

        y_fcn1 = tf.nn.relu(conv2d(atrous_conv2d1, self.W_fcn1) + self.b_fcn1)
        y_fcn2 = conv2d(y_fcn1, self.W_fcn2) + self.b_fcn2
        return tf.reshape(y_fcn2, [-1, 1 * 1 * self.channel])

    def train(self, h, w, X, Y):
        self.h = h
        self.w = w
        self.x_dim = h*w*(self.channel+3)

        x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        y_ = tf.placeholder(tf.float32, shape=[None, self.channel])

        y_resize = self.prenet(x)
        y_out = tf.nn.softmax(y_resize)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())
        acc_pre = 0.
        for i in range(20):
            acc_sum = 0.
            denum = 0.
            N = len(X)
            X = np.array(X, np.float32).reshape(N, self.x_dim)
            Y = np.array(Y, np.float32)
            perm = np.random.permutation(N)
            for j in range(0, N, self.batchsize):
                x_batch = X[perm[j:j + self.batchsize]]
                y_batch = Y[perm[j:j + self.batchsize]]
                acc_sum += sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch})
                denum += 1
                train_step.run(feed_dict={x: x_batch, y_: y_batch})
            acc = acc_sum / denum
            print("step %d,accuracy %g" % (i+1, acc))
            if acc > 0.9999 or (acc - acc_pre) < 0.001:
                break
            acc_pre = acc

    def estimate(self, h, w, X):
        self.h = h
        self.w = w
        self.x_dim = h * w * (self.channel + 3)
        self.y_dim = h * w * self.channel
        x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        y_resize = self.net(x)
        y = tf.nn.softmax(y_resize)
        N = len(X)
        X = np.array(X, np.float32).reshape(N, self.x_dim)
        out = sess.run(y, feed_dict={x: X})
        return out
