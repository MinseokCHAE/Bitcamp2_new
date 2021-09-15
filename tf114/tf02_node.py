import tensorflow as tf

node1 = tf.constant(10.0, tf.float32)
node2 = tf.constant(2.0)

addition = tf.add(node1, node2)
subtraction = tf.subtract(node1, node2)
multiplication = tf.multiply(node1, node2)
division = tf.divide(node1, node2)

print(addition)
# Tensor("Add:0", shape=(), dtype=float32)

#####  session.run #####
session = tf.Session()
print(session.run(addition))
print(session.run(subtraction))
print(session.run(multiplication))
print(session.run(division))
# 12.0
# 8.0
# 20.0
# 5.0

