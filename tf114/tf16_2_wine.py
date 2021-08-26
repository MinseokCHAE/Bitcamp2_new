import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
tf.set_random_seed(21)

datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
y_data = np.array(y_data)
# print(x_data.shape, y_data.shape) # (178, 13) (178,)
# print(np.unique(y_data))    # [0 1 2]
classes = 3

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.int32, shape=[None, ])
y_onehot = tf.one_hot(y, classes)
y_onehot = tf.reshape(y_onehot, [-1, classes])

w = tf.Variable(tf.random_normal([13,classes]))
b = tf.Variable(tf.random_normal([classes]))

logits = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

loss_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot)
loss = tf.reduce_mean(loss_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

prediction = tf.argmax(hypothesis, 1)
target = tf.argmax(y_onehot, 1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epochs in range(2001):
        session.run(train, feed_dict={x:x_data, y:y_data})
        if epochs % 100 == 0:
            loss_val, acc = session.run([loss, accuracy], feed_dict={x:x_data, y:y_data})
            print('epochs = {:5} \tloss = {:.3f} \taccuracy = {:.2%}'.format(epochs,loss_val,acc))

# epochs =  1800  loss = 4612.359         accuracy = 66.29%
# epochs =  1900  loss = 3913.098         accuracy = 65.17%
# epochs =  2000  loss = 1807.328         accuracy = 59.55%

session.close()
