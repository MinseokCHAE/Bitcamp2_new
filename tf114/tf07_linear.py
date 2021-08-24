import tensorflow as tf
tf.set_random_seed(21)

x_train = [1,2,3]
y_train = [3,5,7]

w = tf.Variable([1], dtype=tf.float32) 
b = tf.Variable([1], dtype=tf.float32)  # 초기값 랜덤하게 설정

hypothesis = w * x_train + b
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(2001):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(loss), session.run(w), session.run(b))

