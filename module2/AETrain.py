import tensorflow as tf
import os
# 为显示中文，导入中文字符集
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Autoencoder(object):
    def __init__(self, n_hidden, n_input, learning_rate, scale=0.08):
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
            tf.add(tf.matmul(X + self.training_scale * tf.random_normal((self.n_input,)), self.weights['encoder_h']),
                   self.biases['encoder_b']))
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


X_train = pd.read_csv(os.getcwd() + '\\Data\\faceDataNormal.csv').values

# 创建模型
model = Autoencoder(n_hidden=34, n_input=X_train.shape[1], learning_rate=0.001)

# 定义训练步数、批量大小等超参数
training_epochs = 192
batch_size = 60
display_step = 5
record_step = 5

# 训练模型
total_batch = int(X_train.shape[0] / batch_size)

cost_summary = []

saver = tf.train.Saver()  # 生成saver

print("开始训练")
for epoch in range(training_epochs):
    cost = None
    for i in range(total_batch):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = X_train[batch_start:batch_end, :]

        cost = model.partial_fit(batch)

    if epoch % display_step == 0 or epoch % record_step == 0:
        total_cost = model.calc_total_cost(X_train)

        if epoch % record_step == 0:
            cost_summary.append({'epoch': epoch + 1, 'cost': total_cost})

        if epoch % display_step == 0:
            print("Epoch:{}, cost={:.9f}".format(epoch + 1, total_cost))

save_path = os.getcwd() + '\\model\\'
model.save(save_path + "module2.ckpt")

'''
7．查看迭代步数与损失值的关系
'''
f, ax1 = plt.subplots(1, 1, figsize=(10, 4))
ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(map(lambda x: x['cost'], cost_summary)))
ax1.set_title('损失值', fontproperties=myfont)

plt.xlabel('迭代次数', fontproperties=myfont)
plt.show()