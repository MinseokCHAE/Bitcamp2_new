import tensorflow as tf
tf.set_random_seed(21)

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
x_pred = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 초기값 랜덤하게 설정

hypothesis = w * x_train + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.17338)
train = optimizer.minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(99):
        _, loss_val, w_val, b_val = session.run([train, loss, w, b], 
        feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 5 == 0:
            print(step, loss_val, w_val, b_val)
            # 85 0.00017098208 [1.985498] [1.0254184]
            # 90 9.949916e-05 [1.992394] [1.0223553]
            # 95 5.950498e-05 [1.991091] [1.0168543]

prediction = w_val * x_pred + b_val

session = tf.Session()
session.run(tf.global_variables_initializer())

print(session.run(prediction, feed_dict = {x_pred:[4]}))
print(session.run(prediction, feed_dict = {x_pred:[5,6]}))
print(session.run(prediction, feed_dict = {x_pred:[7,8,9]}))
# [8.992741]
# [10.986987 12.981235]
# [14.975481 16.969728 18.963976]

