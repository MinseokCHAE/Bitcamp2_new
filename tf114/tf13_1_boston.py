from sklearn.datasets import load_boston
import tensorflow as tf
tf.set_random_seed(21)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
# print(x.shape, y.shape) # (506, 13) (506,)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.random_normal([13,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.add(tf.matmul(x, w), b)

loss = tf.reduce_mean(tf.square(y-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(1001):
    loss_val, hypothesis_val, _ = session.run([loss, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, '\n', 'loss = ', loss_val, '\n',  hypothesis_val)

# pred, acc = session.run([prediction, accuracy], feed_dict={x:x_data, y:y_data})
# print('prediction = ', pred, '\n', 'accuracy = ', acc)

session.close()
