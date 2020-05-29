import tensorflow as tf
import os
# 为显示中文，导入中文字符集
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Autoencoder(object):
    def __init__(self, n_hidden, n_input, learning_rate, scale=0.1):
        self.training_scale = scale

        self.n_hidden = n_hidden
        self.n_input = n_input

        self.learning_rate = learning_rate

        self.weights, self.biases = self._initialize_weights()

        self.x = tf.placeholder("float", [None, self.n_input])

        self.encoder_op = self.encoder(self.x)
        self.decoder_op = self.decoder(self.encoder_op)

        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'encoder_h': tf.Variable(tf.random_normal([self.n_input, self.n_hidden]), name='encoder_weight'),
            'decoder_h': tf.Variable(tf.random_normal([self.n_hidden, self.n_input]), name='dncoder_weight'),
        }
        biases = {
            'encoder_b': tf.Variable(tf.random_normal([self.n_hidden]), name='encoder_bias'),
            'decoder_b': tf.Variable(tf.random_normal([self.n_input]), name='dncoder_bias'),
        }
        return weights, biases

    def encoder(self, X):
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(X + self.training_scale * tf.random_normal((self.n_input,), seed=0),
                             self.weights['encoder_h']),
                   self.biases['encoder_b']))

        # + self.training_scale * tf.random_normal((self.n_input,))
        return layer_1

    def decoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h']), self.biases['decoder_b']))
        return layer_1

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.sess.run(self.encoder_op, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoder_op, feed_dict={self.x: X})

    def save(self, model_path):
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)

    def restore(self, model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

save_path = os.getcwd() + '\\model\\'
model = Autoencoder(n_hidden=34, n_input=136, learning_rate=0.001)
model.restore(save_path + "module2.ckpt")


# def getScope(filename):
#     df = pd.read_csv(filename)
#     data60 = df.loc[:60].values
#     data200 = df.loc[60:260].values
#     R1 = np.mean(model.transform(data60), axis=0)
#     R2s = model.transform(data200)
#     means = (R2s - R1) ** 2
#     mse = np.sum(means, axis=1)
#     return min(mse) * 10e6, max(mse) * 10e6

def getScope(filename):
    df = pd.read_csv(filename)
    data200 = df.loc[:200].values
    cost = []
    for i in data200:
        cost.append(model.calc_total_cost(i.reshape((1,-1))))
    return min(cost) * 10e4, max(cost) * 10e4


print(getScope(os.getcwd() + '\\Data\\faceDataNormalTest.csv'))
print(getScope(os.getcwd() + '\\Data\\faceDataSleepy.csv'))
print(getScope(os.getcwd() + '\\Data\\faceDataYawn.csv'))