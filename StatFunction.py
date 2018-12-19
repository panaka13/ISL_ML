import math
import numpy 

def mean(arr):
	return float(sum(arr)) / len(arr)

def dev(x, arr):
	return [(x - y) for y in arr]

def median1(arr):
	arr.sort()
	n = len(arr)
	if n%2:
		return arr[n//2]
	else:
		return (arr[n//2]+arr[n//2-1])/2.0

def median(arr):
	n = len(arr)
	if n%2:
		return getPosition(arr, n/2)
	else:
		return (getPosition(arr, n/2)+getPosition(arr, n/2-1))/2.0

def getPosition(arr, index):
	l = 0
	r = len(arr)-1
	while l <= r:
		x = arr[(l+r)//2]
		i = l
		j = r
		while i <= j:
			while arr[i] < x:
				i += 1
			while arr[j] > x:
				j -= 1
			if i <= j:
				arr[i], arr[j] = arr[j], arr[i]
				i += 1
				j -= 1
		if index <= j:
			r = j
		elif index >= i:
			l = i
		else:
			return x


def sumOfPower(arr, exp):
	return sum([(x**exp) for x in arr]);

def std(arr):
	return math.sqrt(sumOfPower(dev(mean(arr), arr), 2) / len(arr))

def pearson(xs, ys):
	return numpy.corrcoef(xs, ys)[0][1]

def histogram(arr): 
	hist = {}
	for x in arr:
		hist[x] = hist.get(x, 0) + 1
	return hist

def sumOfSquareError(x, y):
	n = len(x)
	z = []
	for i in range(n):
		z.append(x[i]-y[i])
	return math.sqrt(sumOfPower(z, 2)) / math.sqrt(n)

# probability mass function
def pmf(arr): 
	pmf = {}
	for x, freq in histogram(arr).items():
		pmf[x] = freq / len(arr)
	return pmf

# cumulative distribution function
def cdf(arr):
	arr.sort()
	keys = numpy.unique(arr)
	values = []
	cnt = 1.0/len(arr)
	for i in range(1, len(arr)):
		if arr[i] != arr[i-1]:
			values.append(cnt)
		cnt += 1.0/len(arr)
	values.append(cnt)
	return keys, values

# rank
def ranking(arr, x):
	ans = 0
	for y in arr:
		ans += (x >= y)
	return ans

def percentileRank(arr, x):
	return 100 * ranking(arr, x) / len(arr)

def percentileValueSlow(arr, percentile):
	arr.sort()
	for x in arr:
		if percentileRank(arr, x) >= percentile:
			return x

def percentileValue1(arr, percentile):
	arr.sort()
	index = percentile * (len(arr)-1) / 100
	return arr[math.ceil(index)]

def percentileValue(arr, percentile):
	index = math.ceil(percentile * (len(arr)-1) / 100) 
	return getPosition(index)

# plotting
from matplotlib import pyplot

def plotHistogram(hist, discrete=False):  
	keys = list(hist.keys())
	values = list(hist.values())
	if discrete:
		for i in range(len(keys)):
			keys[i] = i+1
	pyplot.bar(keys, values)
	pyplot.show()

def plotHistogramLine(hist):
	pyplot.plot(hist.keys(), hist.values())
	pyplot.show()

def plotHistogramPoint(hist):
	pyplot.plot(hist.keys(), hist.values(), 'ro')
	pyplot.show()

def plotCDF(arr, discrete=False):
	keys, values = cdf(arr)
	if discrete:
		for i in range(len(keys)):
			keys[i] = i+1
	pyplot.plot(keys, values)
	pyplot.show()

def plotCDFs(arrs):
	for arr in arrs:
		keys,values = cdf(arr)
		pyplot.plot(keys, values)
	pyplot.show()

def plotLogScaleY(xs, ys):
	pyplot.plot(xs, ys)
	pyplot.yscale('log')
	pyplot.show()

def plotLogScaleX(xs, ys):
	pyplot.plot(xs, ys)
	pyplot.xscale('log')
	pyplot.show()

def plotXY(xs, ys):
	pyplot.plot(xs, ys)
	pyplot.show()

def meanFromHistogram(hist): 
	return sum([value*freq for value, freq in hist.items()])/sum(hist.values())

def loadXY(fileName):
	import csv
	lines = csv.reader(open(fileName, 'r'))
	x = []
	y = []
	for line in lines:
		x.append(list(map(float, line[:-1])))
		y.append(float(line[-1]))
	return numpy.matrix(x), numpy.matrix(y).transpose()
