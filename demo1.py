import numpy as np
import tensorflow as tf

a = tf.constant(1, shape=(2, 2))

w = tf.Variable(tf.random_normal([2, 3], stddev=2))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])

x = tf.placeholder(tf.float32, name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

biases = tf.Variable(tf.zeros([2, 2]))

sess = tf.InteractiveSession()
# sess.run(w1.initializer)
# sess.run(w2.initializer)
init_op = tf.initialize_all_variables()

sess.run(init_op)
print sess.run(y, feed_dict={x:[[0.7, 0.9]]})
sess.close()
