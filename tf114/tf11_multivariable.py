import tensorflow as tf
tf.set_random_seed(21)

x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b

cost = tf.reduce_mean(tf.square(y-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hypothesis_val, _ = session.run([cost, hypothesis, train], 
                                    feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'cost = ', cost_val, hypothesis_val)

session.close()

#############################################

x_data = [[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]]
y_data = [[152],[185],[180],[205],[142]]

x = tf.placeholder(tf.float32, shape=[None ,3])
y = tf.placeholder(tf.float32, shape=[None ,1])

w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(y-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hypothesis_val, _ = session.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'cost = ', cost_val, hypothesis_val)

session.close()
