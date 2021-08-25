from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.set_random_seed(21)

datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
# print(x_data.shape, y_data.shape) # (442, 10) (442,)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.random_normal([10,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.add(tf.matmul(x, w), b)

loss = tf.reduce_mean(tf.square(y-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(1001):
    loss_val, hypothesis_val, _ = session.run([loss, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, '\n', 'loss = ', loss_val, '\n',  hypothesis_val)

total_error = tf.square(tf.subtract(y, tf.reduce_mean(y)))
unexplained_error = tf.square(tf.subtract(y, hypothesis))
R_squared = tf.reduce_mean(tf.subtract(tf.divide(unexplained_error, total_error), 1.0))
R = tf.multiply(tf.sign(R_squared),tf.sqrt(tf.abs(R_squared)))

total_error, unexplained_error, R_squared, R = session.run(
    [total_error, unexplained_error, R_squared, R], 
    feed_dict={x:x_data, y:y_data})
print('R2 score = ', R/100)
# R2 score =  0.7670832061767578

session.close()
