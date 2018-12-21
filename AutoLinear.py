import pandas as pd
import StatFunction as sf
import math
import numpy as np
import scipy.stats
import Projection
import tensorflow as tf
import time

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


def custom(features = []):
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
	# set up
	n = data.shape[0]
	features.append('free')
	x = data[features].values
	y = data['mpg'].tolist()
	a = Projection.project(x, y)
	for i in range(len(features)):
		coefficient[features[i]] = a[i]
	y_pred = np.dot(x, a)
	# std_error
	y_mean = sf.mean(y)
	rss = sf.sumOfPower([y[i]-y_pred[i] for i in range(n)], 2)
	rse = math.sqrt(rss/(n-len(features)))
	print(rse)
	for feature in features:
		if feature != 'free':
			x = data[feature].values
			mean = sf.mean(x)
			std = sf.sumOfPower([mean-i for i in x], 2)
			std_error[feature] = rse/math.sqrt(std)
	# t_score (0, n)
	for feature in features:
		if feature != 'free':
			t_score[feature] = coefficient[feature]/std_error[feature]
	# p_value
	for feature in features:
		if feature != 'free':
			p_value[feature] = (1-scipy.stats.t.cdf(abs(t_score[feature]), n-2))*2
	print(coefficient)
	print(std_error)
	print(t_score)
	print(p_value)

def tensorflow_read(features = []):
	tmpx, tmpy = readData(features)
	x = tf.data.Dataset.from_tensor_slices(tmpx)
	y = tf.data.Dataset.from_tensor_slices(tmpy)
	return (x, y, len(tmpy), len(tmpx[0]))

def tensorflow_train(features = []):
	# parameter
	iteration = num_run
	learning_rate = 0.0000001
	display_step = 1
	# get data
	x_train, y_train = readData(features)
	n = len(x_train)
	m = len(x_train[0])
	# set up placeholder
	x = tf.placeholder(tf.float32, (m))
	y = tf.placeholder(tf.float32)
	# set up model
	a = tf.Variable(tf.zeros([m, 1]), name="weight")
	y_pred = tf.math.reduce_sum(tf.multiply(a, x), keepdims=True)
	cost = tf.reduce_sum(tf.math.square(y_pred - y))/n
	optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	test = np.identity(m)
	# init
	init = tf.global_variables_initializer()
	# start train
	with tf.Session() as sess:
		# run init
		sess.run(init)
		for i in range(m):
			c = sess.run(cost, feed_dict = {x: test[i], y: 0})
			print(c)
		# training
		for step in range(iteration):
			for i in range(n):
				sess.run(optimize, feed_dict = {x: x_train[i], y: y_train[i]})
			# display
			if (step+1) % display_step == 0:
				s = 0
				for i in range(n):
					s += sess.run(cost, feed_dict = {x: x_train[i], y: y_train[i]})
					if (i == 0):
						print("x: {0}".format(sess.run(x, feed_dict = {x: x_train[i], y: y_train[i]})))
						print("y: {0}".format(sess.run(y, feed_dict = {x: x_train[i], y: y_train[i]})))
						print("a: {0}".format(sess.run(a, feed_dict = {x: x_train[i], y: y_train[i]})))
						print("y_pred: {0}".format(sess.run(y_pred, feed_dict = {x: x_train[i], y: y_train[i]})))
						print("minus: {0}".format(sess.run(y_pred-y, feed_dict = {x: x_train[i], y: y_train[i]})))
				print("Iteration {0}, cost: {1}".format(step+1, s))

def tensorflow_train_batch(features = []):
	# parameter
	iteration = num_run
	learning_rate = 0.0000001
	display_step = 1
	# get data
	x_train, y_train, n, m = tensorflow_read(features)
	data = tf.data.Dataset.zip((x_train, y_train))
	data = data.batch(392)
	iterator = data.make_initializable_iterator()
	next_element = iterator.get_next()
	# set up placeholder
	x = tf.placeholder(tf.float32, ([None, m]), name="X")
	y = tf.placeholder(tf.float32, [None], name="Y")
	batch_size = tf.placeholder(tf.int32, name="size")
	new_y = tf.reshape(y, [batch_size, 1])
	# set up model
	a = tf.Variable(tf.zeros([m, 1]), name="weight")
	y_pred = tf.matmul(x, a)
	cost = tf.reduce_sum(tf.math.square(y_pred - new_y)) / tf.to_float(batch_size)
	optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	# init
	init = tf.global_variables_initializer()
	test = np.identity(m)
	# start train
	with tf.Session() as sess:
		# run init
		sess.run(init)
		for i in range(m):
			c = sess.run(cost, feed_dict = {x: [test[i]], y: [0], batch_size: 1})
			print(c)
		# training
		partial_time = time.monotonic()
		for step in range(iteration):
			sess.run(iterator.initializer)
			try:
				while True:
					(xx, yy) = sess.run(next_element)
					sess.run(optimize, feed_dict = {x: xx, y: yy, batch_size: len(yy)})
			except tf.errors.OutOfRangeError:
				pass
			# display
			if (step+1) % display_step == 0:
				sess.run(iterator.initializer)
				loss = 0
				try:
					while True:
						(xx, yy) = sess.run(next_element)
						c = sess.run(cost, feed_dict = {x: xx, y: yy, batch_size: len(yy)})
						loss += c * len(yy)
				except tf.errors.OutOfRangeError:
					pass
				print("Iteration {0}, cost: {1}".format(step+1, loss/n))
				print("Running time: {0}".format(time.monotonic() - partial_time))
				partial_time = time.monotonic()


num_run = 1

start_time = time.monotonic()
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']
tensorflow_train_batch(features)
print(time.monotonic() - start_time)
#start_time = time.monotonic()
#features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']
#tensorflow_train(features)
#print(time.monotonic() - start_time)
