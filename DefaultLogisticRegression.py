import numpy as np
import pandas as pd
import StatFunction as sf
import math
import scipy.stats
import Projection
import tensorflow as tf

def getData(predicts = 'default', features = []):
	data = pd.read_csv('Default.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	# convert data
	data['student'] = (data['student'] == 'Yes')*1
	data['default'] = (data['default'] == 'Yes')*1
	features.append('free')
	print(features)
	x = data[features].values
	y = data[predicts].values
	n = len(x)
	m = len(x[0])
	x = tf.data.Dataset.from_tensor_slices(x)
	y = tf.data.Dataset.from_tensor_slices(y)
	return (x, y, n, m)

def train(predicts='default', features=[]):
	# parameter
	iteration = 1000000
	display = 100
	learning_rate = 0.000001
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
	a = tf.Variable(tf.zeros([m, 1]), name="a")
	fx = tf.matmul(x, a)
	px = 1/(1 + tf.exp(-fx))
	loss = tf.reduce_sum(-tf.log((1-y_label)*(1-px) + px*y_label)) / tf.to_float(batch_size)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(a))
		for iter_cnt in range(iteration):
			try:
				sess.run(iterator.initializer)
				while True:
					(xx, yy) = sess.run(next_batch)
					nn = len(yy)
					#debug = {x: xx, y: yy, batch_size: nn}
					#px_value = sess.run(px, feed_dict = debug)
					#print("px: {0}".format(px_value))
					#print("fx: {0}".format(sess.run(fx, feed_dict = debug)))
					#print("px*y: {0}".format(sess.run(px*y_label, feed_dict = debug)))
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
				print(sess.run(a))


features = ['balance']
train(features = features)
