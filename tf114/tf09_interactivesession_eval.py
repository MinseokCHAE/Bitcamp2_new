import tensorflow as tf
tf.set_random_seed(21)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 

session = tf.Session()
session.run(tf.global_variables_initializer())
# print(session.run(w)) # [-0.43087453]
session.close()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
# print(w.eval())   # [-0.43087453]
session.close()

session = tf.Session()
session.run(tf.global_variables_initializer())
# print(w.eval(session=session))  # [-0.43087453]
session.close()


x = [1, 2, 3]
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)
hypothesis = w * x + b

session = tf.Session()
session.run(tf.global_variables_initializer())
# print(session.run(hypothesis))  # [1.3       1.6       1.9000001]
session.close()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
# print(hypothesis.eval())  # [1.3       1.6       1.9000001]
session.close()

session = tf.Session()
session.run(tf.global_variables_initializer())
# print(hypothesis.eval(session=session))  # [1.3       1.6       1.9000001]
session.close()
