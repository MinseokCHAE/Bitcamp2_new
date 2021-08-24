import tensorflow as tf
print(tf.__version__)

# print('hello')
hello = tf.constant('hello')
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

session = tf.Session()
print(session.run(hello))
# b'hello'