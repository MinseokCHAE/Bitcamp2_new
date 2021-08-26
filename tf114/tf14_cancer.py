import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
tf.set_random_seed(21)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = np.array(y_data)
# print(x_data.shape, y_data.shape) # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.random_normal([30,1]))
b = tf.Variable(tf.random_normal([1]))

logits = tf.matmul(x, w) + b
hypothesis = tf.sigmoid(logits)

loss = tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.11)
train = optimizer.minimize(loss)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epochs in range(2001):
        session.run(train, feed_dict={x:x_data, y:y_data})
        if epochs % 100 == 0:
            loss_val, acc = session.run([loss, accuracy], feed_dict={x:x_data, y:y_data})
            print('epochs = {:5} \tloss = {:.3f} \taccuracy = {:.2%}'.format(epochs,loss_val,acc))

# epochs =  1800  loss = nan      accuracy = 37.26%
# epochs =  1900  loss = nan      accuracy = 37.26%
# epochs =  2000  loss = nan      accuracy = 37.26%

session.close()
