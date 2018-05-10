from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy.random

sess = tf.InteractiveSession()
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class DNN(object):

    def __init__(self, x_dim=243, y_dim=3, patch_radius=4, channel=3):
        self.batchsize = 20
        self.patch_radius = patch_radius
        self.channel = channel
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.W_conv1 = weight_variable([3, 3,  self.channel, 64])
        self.b_conv1 = bias_variable([64])
        self.W_conv2 = weight_variable([3, 3, 64, 128])
        self.b_conv2 = bias_variable([128])
        self.W_conv3 = weight_variable([3, 3, 128, 256])
        self.b_conv3 = bias_variable([256])
        self.W_fc1 = weight_variable([1 * 1 * 256, 256])
        self. b_fc1 = bias_variable([256])
        self.W_fc2 = weight_variable([1 * 1 * 256, y_dim])
        self.b_fc2 = bias_variable([y_dim])

    def net(self, X):
        x_image = tf.reshape(
            X, [-1, self.patch_radius * 2 + 1, self.patch_radius * 2 + 1, self.channel])
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_pool2, self.W_conv3) + self.b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1 * 1 * 256])
        y_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
        # h_fc1_drop = tf.nn.dropout(y_fc1, 0.5)
        y_fc2 = tf.matmul(y_fc1, self.W_fc2) + self.b_fc2
        return y_fc2


    def train(self, X, Y):
        x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        y_ = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        y_fc2 = self.net(x)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc2))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_fc2, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())
        acc_pre = 0.
        for i in range(40):
            acc_sum = 0.
            denum = 0.
            N = len(X)
            X = numpy.array(X, numpy.float32).reshape(N, self.x_dim)
            Y = numpy.array(Y, numpy.float32)
            perm = numpy.random.permutation(N)
            for j in range(0, N, self.batchsize):
                x_batch = X[perm[j:j + self.batchsize]]
                y_batch = Y[perm[j:j + self.batchsize]]
                acc_sum += sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch})
                denum += 1
                train_step.run(feed_dict={x: x_batch, y_: y_batch})
            acc = acc_sum / denum
            print("step %d,accuracy %g" % (i+1, acc))
            if(acc > 0.999 or (acc_pre - acc) > 0.01):
                break
            acc_pre = acc

    def estimate(self, X):
        out = list()
        X = numpy.array(X, numpy.float32).reshape(len(X), self.x_dim)
        for j in range(0, len(X), 10000):
            x_batch = X[j:j + 10000]
            x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            y_fc2 = self.net(x)
            y = tf.nn.softmax(y_fc2)
            y_ = sess.run(y, feed_dict={x: x_batch})
            out.extend(y_)
        return out
