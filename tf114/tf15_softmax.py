import numpy as np
import tensorflow as tf
tf.set_random_seed(21)

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,6,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4,3]))
b = tf.Variable(tf.random_normal([1, 3]))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

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

prediction = session.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
print('prediction = ', session.run(tf.math.argmax(prediction)))
# prediction =  [0 0 0]

session.close()
