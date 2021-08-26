import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
tf.set_random_seed(21)

datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
# print(x_data.shape, y_data.shape) # (150, 4) (150,)
# print(np.unique(y_data)) # [0 1 2]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.int32, shape=[None, 1])
y = tf.one_hot(y, 3)
y = tf.reshape(y, [-1, 3])

w = tf.Variable(tf.random_normal([4,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))   # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(1001):
    loss_val, hypothesis_val, _ = session.run([loss, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'loss = ', loss_val, '\n',  hypothesis_val)

prediction = tf.argmax(hypothesis, 1)
target = tf.argmax(y, 1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: %.2f' % session.run(accuracy * 100, feed_dict={x: x_data, y: y_data}))

session.close()
