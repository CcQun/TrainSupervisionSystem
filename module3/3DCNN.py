import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import DataSet


def CNN_3d(x, out_channels_0, out_channels_1, out_channels_2, add_relu=True):
    '''Add a 3d convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_depth, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[3, 3, 3, in_channels, out_channels_0],
        # dtype=tf.float64,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        # dtype=tf.float64,
        initializer=tf.zeros_initializer())

    conv_0 = tf.nn.conv3d(tf.cast(x, tf.float32), weights_0, strides=[1, 1, 1, 1, 1], padding="SAME")
    print('conv_0 shape: %s' % conv_0.shape)
    conv_0 = conv_0 + bias_0

    if add_relu:
        conv_0 = tf.nn.relu(conv_0)

    pooling_0 = tf.nn.max_pool3d(
        conv_0,
        ksize=[1, 2, 2, 1, 1],
        strides=[1, 2, 2, 1, 1],
        padding="SAME")
    print('pooling_0 shape: %s' % pooling_0.shape)

    # layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[3, 3, 3, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_1 = tf.nn.conv3d(pooling_0, weights_1, strides=[1, 1, 1, 1, 1], padding="SAME")
    print('conv_1 shape: %s' % conv_1.shape)
    conv_1 = conv_1 + bias_1

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool3d(
        conv_1,
        ksize=[1, 2, 2, 2, 1],
        strides=[1, 2, 2, 2, 1],
        padding="SAME")
    print('pooling_1 shape: %s' % pooling_1.shape)

    # layer_2
    weights_2 = tf.get_variable(
        name='filter_2',
        shape=[3, 3, 3, out_channels_1, out_channels_2],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_2 = tf.get_variable(
        name='bias_2',
        shape=[out_channels_2],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_2 = tf.nn.conv3d(pooling_1, weights_2, strides=[1, 1, 1, 1, 1], padding="SAME")
    print('conv_1 shape: %s' % conv_2.shape)
    conv_2 = conv_2 + bias_2

    if add_relu:
        conv_2 = tf.nn.elu(conv_2)

    pooling_2 = tf.nn.max_pool3d(
        conv_2,
        ksize=[1, 2, 2, 2, 1],
        strides=[1, 2, 2, 2, 1],
        padding="SAME")
    print('pooling_2 shape: %s' % pooling_2.shape)

    return tf.contrib.layers.flatten(pooling_2)


data, label = DataSet.dataset()

num_classes = 8

label = np.eye(num_classes)[label.reshape(-1).astype(np.int8)]

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0, test_size=0.2)


X = tf.placeholder(tf.float32, shape=(None, 128, 128, 15, 1))
y = tf.placeholder(tf.float32, shape=(None, num_classes))

flatten0 = CNN_3d(X_train, 32, 64, 128)

print(flatten0.shape)

num1 = int(flatten0.shape[1])
weight0 = tf.Variable(tf.truncated_normal([num1, 128]))
bias0 = tf.Variable(tf.constant(0.1, shape=[128]))

flatten1 = tf.nn.sigmoid(tf.add(tf.matmul(flatten0, weight0), bias0))

num2 = int(flatten1.shape[1])
weight1 = tf.Variable(tf.truncated_normal([num2, num_classes]))
bias1 = tf.Variable(tf.constant(0.1, shape=[num_classes]))

output = tf.nn.softmax(tf.add(tf.matmul(flatten1, weight1), bias1))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, float))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)

epoch_num = 150
batch_size = 128
total_batch = int(X_train.shape[0] / batch_size)
display_step = 10
record_step = 5

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_summary = []
for epoch_num in range(epoch_num):
    for batch_num in range(total_batch):
        batch_start = batch_num * batch_size
        batch_end = (batch_num + 1) * batch_size
        train_data = X_train[batch_start:batch_end, :]
        train_label = y_train[batch_start:batch_end, :]
        cost, opt = sess.run((loss, optimizer), feed_dict={X: train_data, y: train_label})

    if epoch_num % display_step == 0 or epoch_num % record_step == 0:
        total_cost, total_accuracy = sess.run((loss, accuracy), feed_dict={X: X_train, y: y_train})
        if epoch_num % display_step == 0:
            total_test_cost, total_test_accuracy = sess.run((loss, accuracy), feed_dict={X: X_test, y: y_test})
            print('Epoch{}:'.format(epoch_num + 1))
            print('    Train:cost={:.9f},accuracy={:.5f}'.format(total_cost, total_accuracy))
            print('    Test:cost={:.9f},accuracy={:.5f}'.format(total_test_cost, total_test_accuracy))
        if epoch_num % record_step == 0:
            cost_summary.append({'epoch': epoch_num + 1, 'cost': total_cost})

f, ax1 = plt.subplots(1, 1, figsize=(10, 4))
ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(map(lambda x: x['cost'], cost_summary)))
ax1.set_title('cost')

plt.xlabel('epoch num')
plt.show()

saver = tf.train.Saver()
save_path = os.getcwd() + '\\model\\'
saver.save(sess,save_path + "module3.ckpt")