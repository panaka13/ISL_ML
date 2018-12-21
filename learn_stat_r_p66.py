import pandas as pd
import StatFunction as sf
import math
import Projection
import numpy as np
import scipy.stats

def twofeature():
	data = pd.read_csv('Advertising.csv')
	y = data['sales'].tolist()
	meany = sf.mean(y)
	x = data['TV'].tolist()
	meanx = sf.mean(x)
	n = len(x)

# y_ = ax_ + b + error
#calculate a
	den = 0
	num = 0
	for i in range(len(x)):
		den += (x[i]-meanx)**2
		num += (x[i]-meanx)*(y[i]-meany)
	a = num/den
#calculate b 
	b = meany - meanx*a
	f = lambda x: x*a+b

# error term
	rss = sf.sumOfPower([y[i]-f(x[i]) for i in range(len(x))], 2)
	rse = math.sqrt(rss/(n-2))
	print(rse)
	stdA = math.sqrt(rse**2 / den)
	stdB = rse * math.sqrt(1/n + (meanx**2)/(den))

# t-statistic
	tscore_a = a/stdA
	tscore_b = b/stdB

# R^2
	tss = sf.sumOfPower([y[i]-meany for i in range(n)], 2)
	R = math.sqrt(1-rss/tss)
	print(R**2)

def allfeature():
	data = pd.read_csv('Advertising.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	y = data['sales'].tolist()
	x = data[['free', 'TV', 'radio', 'newspaper']].values
	n = len(y)
	a = Projection.project(x, y)
	print(a)
	y_pred = np.dot(x, a)
	# error term
	rss = sf.sumOfPower([y[i]-y_pred[i] for i in range(n)], 2)
	rse = math.sqrt(rss/(n-2))
	# std of feature
	mean_tv = sf.mean(data['TV'].values)
	rss_tv = sf.sumOfPower([x-mean_tv for x in data['TV'].values], 2)
	se_tv = rse / math.sqrt(rss_tv)
	mean_radio = sf.mean(data['radio'].values)
	rss_radio = sf.sumOfPower([x-mean_radio for x in data['radio'].values], 2)
	se_radio = rse / math.sqrt(rss_radio)
	mean_newspaper = sf.mean(data['newspaper'])
	rss_newspaper = sf.sumOfPower([x-mean_newspaper for x in data['newspaper'].values], 2)
	se_newspaper = rse / math.sqrt(rss_newspaper)
	print("TV std_error: {0}".format(se_tv))
	print("radio std_error: {0}".format(se_radio))
	print("newspaper std_error: {0}".format(se_newspaper))

def custom(features = []):
	data = pd.read_csv('Advertising.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	y = data['sales'].tolist()
	features.append('free')
	x = data[features].values
	n = len(y)
	a = Projection.project(x, y)
	coefficient = {}
	for i in range(len(features)):
		coefficient[features[i]] = a[i]
	print(coefficient)
	y_pred = np.dot(x, a)
	# error term
	rss = sf.sumOfPower([y[i]-y_pred[i] for i in range(n)], 2)
	print("rss: {0}".format(rss))
	rse = math.sqrt(rss/(n-len(features)))
	std_error = {}
	# std error of feature
	for feature in features:
		if feature != 'free':
			mean = sf.mean(data[feature].values)
			rss = sf.sumOfPower([x-mean for x in data[feature].values], 2)
			se = rse / math.sqrt(rss)
			std_error[feature] = se
	print("std_error: {0}".format(std_error))
	# t-statistic
	t_score = {}
	for feature in features:
		if feature != 'free':
			t_score[feature] = coefficient[feature] / std_error[feature]
	print("t_score: {0}".format(t_score))
	# p-value 
	p_value = {}
	for feature in features:
		if feature != 'free':
			p_value[feature] = (1-scipy.stats.t.cdf(abs(t_score[feature]), n-2))*2
	print("p_value: {0}".format(p_value))
	print(n)

def custom(features = []):
	data = pd.read_csv('Advertising.csv')
	data['free'] = pd.Series(np.ones(data.shape[0]))
	y = data['sales'].tolist()
	features.append('free')
	x = data[features].values
	n = len(y)
	a = Projection.project(x, y)
	coefficient = {}
	for i in range(len(features)):
		coefficient[features[i]] = a[i]
	print(coefficient)
	y_pred = np.dot(x, a)
	# error term
	rss = sf.sumOfPower([y[i]-y_pred[i] for i in range(n)], 2)
	print("rss: {0}".format(rss))
	rse = math.sqrt(rss/(n-len(features)))
	std_error = {}
	# std error of feature
	for feature in features:
		if feature != 'free':
			mean = sf.mean(data[feature].values)
			rss = sf.sumOfPower([x-mean for x in data[feature].values], 2)
			se = rse / math.sqrt(rss)
			std_error[feature] = se
	print("std_error: {0}".format(std_error))
	# t-statistic
	t_score = {}
	for feature in features:
		if feature != 'free':
			t_score[feature] = coefficient[feature] / std_error[feature]
	print("t_score: {0}".format(t_score))
	# p-value 
	p_value = {}
	for feature in features:
		if feature != 'free':
			p_value[feature] = (1-scipy.stats.t.cdf(abs(t_score[feature]), n-2))*2
	print("p_value: {0}".format(p_value))
	print(n)
