import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(21)

mnist = input_data.read_data_sets('../data/study/mnist_data', one_hot=True)
classes = 10

x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, classes])

# w1 = tf.Variable(tf.random_normal([28*28,16]))
# b1 = tf.Variable(tf.random_normal([16]))
# layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# w2 = tf.Variable(tf.random_normal([16,64]))
# b2 = tf.Variable(tf.random_normal([64]))
# layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

# w3 = tf.Variable(tf.random_normal([64,32]))
# b3 = tf.Variable(tf.random_normal([32]))
# layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

# w4 = tf.Variable(tf.random_normal([32,8]))
# b4 = tf.Variable(tf.random_normal([8]))
# layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)

# w5 = tf.Variable(tf.random_normal([8,4]))
# b5 = tf.Variable(tf.random_normal([4]))
# layer5 = tf.nn.relu(tf.matmul(layer4, w5) + b5)

w6 = tf.Variable(tf.random_normal([28*28,classes]))
b6 = tf.Variable(tf.random_normal([classes]))
logits = tf.matmul(x, w6) + b6
hypothesis = tf.nn.softmax(logits)

loss_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(loss_i)
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.543)
train = optimizer.minimize(loss)

prediction = tf.argmax(hypothesis, 1)
# target = tf.argmax(y_onehot, 1)
target = tf.argmax(y, 1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 21
batch_size = 21

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epochs in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            loss_val, _ = session.run([loss, train], feed_dict={x:batch_x, y:batch_y})
            avg_loss += loss_val / total_batch
        print('epoch = ', '%04d' % (epochs + 1), 'loss = ', '{:.9f}'.format(avg_loss))

    print('accuracy = ', accuracy.eval(session=session,
                                        feed_dict={x:mnist.test.images, y:mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print('label = ', session.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('prediction = ', session.run(tf.argmax(hypothesis, 1),
                                        feed_dict={x:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()

# epoch =  0018 loss =  0.325874007
# epoch =  0019 loss =  0.325059801
# epoch =  0020 loss =  0.325965589
# epoch =  0021 loss =  0.326020300
# accuracy =  0.9168
# label =  [1]
# prediction =  [1]

session.close()
