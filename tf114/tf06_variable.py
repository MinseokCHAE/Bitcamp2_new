import tensorflow as tf

session = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

initialize = tf.global_variables_initializer()
session.run(initialize)

print(session.run(x))
