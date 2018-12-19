import tensorflow as tf

x = tf.constant([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=tf.float32)
print(x.shape)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1, use_bias=False)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(loss))
for i in range(100):
  _, loss_value = sess.run((train, loss))

print(sess.run(loss))
