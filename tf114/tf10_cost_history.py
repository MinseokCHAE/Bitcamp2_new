import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [2., 4., 6.]
w = tf.placeholder(tf.float32)
hypothesis = w * x 

cost = tf.reduce_mean(tf.square(y-hypothesis))
w_history = []
cost_history = []

with tf.Session() as session:
    for i in range(-30, 50):
        w_current = i
        cost_current = session.run(cost, feed_dict={w:w_current})

        w_history.append(w_current)
        cost_history.append(cost_current)

plt.plot(w_history, cost_history)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()
