import numpy as np
import tensorflow as tf
tf.set_random_seed(21)

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 16
batch_size = 128
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.get_variable('w1', shape=[3,3,1,32]) # [kernel_size, input, output]
# w1 = tf.Variable(tf.random_normal9[3,3,1,32], dtype=dt.float32)
# w1 = tf.Variable([1], dtype=tf.float32)
layer1 = tf.nn.con2d(x, w1, strides=[1,1,1,1], padding='SAME') # w1 shape(4차원)에 맞추기
layer1 = tn.nn.relu(layer1)
layer1 = tf.nn.max_pool(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w2 = tf.get_variable('w2', shape=[3,3,32,64])
layer2 = tf.nn.con2d(layer1, w2, strides=[1,1,1,1], padding='SAME')
layer2 = tf.nn.relu(layer2)
layer2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

'''
input = Input((28,28,1))
xx = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(input)
xx = MaxPool2D()(xx)
xx = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(xx)
xx = MaxPool2D()(xx)

이후 flatten, dnn, softmax 까지 진행

'''
