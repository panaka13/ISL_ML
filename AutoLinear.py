import pandas as pd
import StatFunction as sf
import math
import numpy as np
import scipy.stats
import Projection

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
	# print(coefficient)
	# print(std_error)
	# print(t_score)
	print(p_value)


features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']
custom(features)
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'year', 'origin']
custom(features)
