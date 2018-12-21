import numpy as np
import pandas as pd
import tensorflow as tf
import time

start_time = time.monotonic()

def readData(features = []):
	data = pd.read_csv('Auto.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	coefficient = {}
	std_error = {}
	t_score = {}
	p_value = {}
	# convert data
	for feature in features:
		if data[feature].dtype == 'object':
			to_drop = []
			for index, row in data.iterrows():
				if row[feature] == '?':
					to_drop.append(index)
			data = data.drop(to_drop, 0)
	for column in data.columns:
		try:
			data[column] = data[column].astype('float')
		except ValueError:
			print("Cant convert column {0} to number".format(column))
	x = data[features].values
	y = data['mpg'].tolist()
	return (x, y)


sess = tf.Session()

data = pd.read_csv("Auto.csv")
data['free'] = pd.Series(np.ones(data.shape[0]))
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']
tmpx, tmpy = readData(features)
real_y = tf.constant(list(map(lambda x: [x], tmpy)))
x = tf.constant(tmpx)
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
optimizer = tf.train.GradientDescentOptimizer(0.0000001)
train = optimizer.minimize(loss)

# training
# for i in range(100):
#	_, loss_value = sess.run((train, loss))
#	print(loss)

partial = time.monotonic()
try:
	i = 0
	while True:
		sess.run(train)
		i += 1
		if i % 10000 == 0:
			print("Step {1} lost: {0}".format(sess.run(loss), i))
			print("running time: {0}".format(time.monotonic() - partial))
			partial = time.monotonic()
except KeyboardInterrupt:
	print("Step {1} lost: {0}".format(sess.run(loss), i))

print("Lost: {0}".format(sess.run(loss)))
print(time.monotonic() - start_time)
