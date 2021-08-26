import numpy as np
import tensorflow as tf
tf.set_random_seed(21)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2,32]))
b1 = tf.Variable(tf.random_normal([32]))
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([32,128]))
b2 = tf.Variable(tf.random_normal([128]))
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([128,64]))
b3 = tf.Variable(tf.random_normal([64]))
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([64,64]))
b4 = tf.Variable(tf.random_normal([64]))
layer4 = tf.sigmoid(tf.matmul(layer3, w4) + b4)

w5 = tf.Variable(tf.random_normal([64,32]))
b5 = tf.Variable(tf.random_normal([32]))
layer5 = tf.sigmoid(tf.matmul(layer4, w5) + b5)

w6 = tf.Variable(tf.random_normal([32,1]))
b6 = tf.Variable(tf.random_normal([1]))

logits = tf.matmul(layer5, w6) + b6
hypothesis = tf.sigmoid(logits)

loss = tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
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

# epochs =  1800  loss = -0.721   accuracy = 50.00%
# epochs =  1900  loss = -0.721   accuracy = 50.00%
# epochs =  2000  loss = -0.721   accuracy = 50.00%

session.close()
