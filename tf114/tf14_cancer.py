from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(21)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
# print(x_data.shape, y_data.shape) # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.random_normal([30,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

cost = tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))    # binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01234)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(1001):
    cost_val, hypothesis_val, _ = session.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'cost = ', cost_val, '\n',  hypothesis_val)

pred, acc = session.run([prediction, accuracy], feed_dict={x:x_data, y:y_data})
print('prediction = ', pred, '\n', 'accuracy = ', acc)
#  accuracy =  0.37258348

session.close()
