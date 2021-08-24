import tensorflow as tf
tf.set_random_seed(21)

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 초기값 랜덤하게 설정

hypothesis = w * x_train + b
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1111)
train = optimizer.minimize(loss)

# session = tf.Session()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(99):
        _, loss_val, w_val, b_val = session.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 5 == 0:
            print(step, loss_val, w_val, b_val)

