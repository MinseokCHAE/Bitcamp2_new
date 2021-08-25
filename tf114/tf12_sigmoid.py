import tensorflow as tf
tf.set_random_seed(21)

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

cost = tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))    # binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hypothesis_val, _ = session.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'cost = ', cost_val, '\n',  hypothesis_val)

pred, acc = session.run([prediction, accuracy], feed_dict={x:x_data, y:y_data})
print('prediction = ', pred, '\n', 'accuracy = ', acc)

session.close()
