import numpy as np
import pandas as pd
import tensorflow as tf

sess = tf.Session()

data = pd.read_csv("Advertising.csv")
data['free'] = pd.Series(np.ones(data.shape[0]))
real_y = tf.constant(list(map(lambda x: [x], data['sales'])))
x = tf.constant(data[['free', 'newspaper']].values)
print(x.shape)

# create model
linear_model = tf.layers.Dense(units=1, use_bias=False)
y = linear_model(x)

# initial weights, etc... to be default value
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model.kernel))

# loss function
loss = tf.losses.mean_squared_error(labels = real_y, predictions = y)
print(sess.run(loss))
 
# gradient descient
optimizer = tf.train.GradientDescentOptimizer(0.000010)
train = optimizer.minimize(loss)

# training
# for i in range(100):
#	_, loss_value = sess.run((train, loss))
#	print(loss)

try:
	i = 0
	while True:
		sess.run(train)
		if i % 10000 == 0:
			print(sess.run(loss))
			print(sess.run(linear_model.kernel))
		i += 1
except KeyboardInterrupt:
	print(sess.run(loss))
	print(sess.run(linear_model.kernel))
