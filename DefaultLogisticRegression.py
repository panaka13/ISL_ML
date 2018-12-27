import numpy as np
import pandas as pd
import StatFunction as sf
import math
import scipy.stats
import Projection
import tensorflow as tf

def getDataRaw(predicts = 'default', features = [], frac = 1, equal = False):
	data = pd.read_csv('Default.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	if equal:
		dfTrue = data[data[predicts] == "Yes"]
		dfFalse = data[data[predicts] == "No"]
		size = min(dfTrue.shape[0], dfFalse.shape[0])
		data = dfTrue.sample(n = size).append(dfFalse.sample(n = size))
	data = data.sample(frac = frac)
	print(data)
	# convert data
	data['student'] = (data['student'] == 'Yes')*1
	data['default'] = (data['default'] == 'Yes')*1
	features.append('free')
	print(features)
	x = data[features].values
	y = data[predicts].values
	return (x, y)

def getData(predicts = 'default', features = []):
	x, y = getDataRaw(predicts, features, equal = True)
	print(x[0])
	n = len(x)
	m = len(x[0])
	x = tf.data.Dataset.from_tensor_slices(x)
	y = tf.data.Dataset.from_tensor_slices(y)
	return (x, y, n, m)

def predicts(param, xx):
	x = tf.placeholder(tf.float32, [None, len(xx)], "X")
	a = param
	fx = tf.matmul(x, a)
	px = 1/(1 + tf.exp(-fx))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		y = sess.run(px, feed_dict = {x: [xx]})
		if y<0.5:
			return 0
		else:
			return 1

def train(predicts='default', features=[]):
	# parameter
	iteration = 10000
	display = 100
	learning_rate = 0.00001
	nsize = 1000
	# set up data
	(xtrain, ytrain, n, m) = getData(predicts, features)
	data = tf.data.Dataset.zip((xtrain, ytrain)).batch(nsize)
	iterator = data.make_initializable_iterator()
	next_batch = iterator.get_next()
	batch_size = tf.placeholder(tf.int32)
	# set up model
	x = tf.placeholder(tf.float32, [None, m], "X")
	y = tf.placeholder(tf.float32, name="Y")
	y_label = tf.reshape(y, [batch_size, 1])
	a = tf.Variable(tf.random.uniform([2, 1]), name="a")
	fx = tf.matmul(x, a)
	px = 1/(1 + tf.exp(-fx))
	loss = tf.reduce_sum(-tf.log((1-y_label)*(1-px) + px*y_label)) / tf.to_float(batch_size)
	gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#print(sess.run(a))
		for iter_cnt in range(iteration):
			try:
				sess.run(iterator.initializer)
				while True:
					(xx, yy) = sess.run(next_batch)
					nn = len(yy)
					debug = {x: xx, y: yy, batch_size: nn}
					#px_value = sess.run(px, feed_dict = debug)
					#print("px: {0}".format(px_value))
					#print("fx: {0}".format(sess.run(fx, feed_dict = debug)))
					#print("px*y: {0}".format(sess.run(px*y_label, feed_dict = debug)))
					gradient = sess.run(gradient_descent, feed_dict = debug)
					#print(gradient)
					sess.run(optimizer, feed_dict = {x: xx, y: yy, batch_size: nn})
			except tf.errors.OutOfRangeError:
				pass
			if (iter_cnt+1) % display == 0:
				sess.run(iterator.initializer)
				total = 0
				try:
					sess.run(iterator.initializer)
					while True:
						(xx, yy) = sess.run(next_batch)
						nn = len(yy)
						tmp = sess.run(loss, feed_dict = {x: xx, y: yy, batch_size: nn})
						total += tmp*nn / n
				except tf.errors.OutOfRangeError:
					pass
				print("Iteration {0} loss: {1}".format(iter_cnt+1, total))
				#myfile.write("{0}\n".format(total))
				aa = sess.run(a)
				#print(aa)
		return (a, sess.run(a))

#myfile = open("result.txt", "w")
features = ['student']
try:
	#myfile.write("{0}".format(train(features = features)))
	param, a = train(features = features)
	print(param)
	features = ['student']
	(x, y) = getDataRaw(features = features, equal = True)
	cnt = 0
	for i in range(len(y)):
		pred_y = predicts(param, x[i])
		cnt += pred_y == y[i]
		print("{0}: pred {1} real {2}".format(i, pred_y, y[i]))
	print(cnt)
except KeyboardInterrupt:
	pass
#myfile.close()
