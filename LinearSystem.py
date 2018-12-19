import numpy
import copy

eps = 1e-15

def multiply(a, k):
	return [x*k for x in a]

def add(a, b):
	return [a[i]+b[i] for i in range(len(a))]

def printing(a):
	for i in range(len(a)):
			print('{0:.15f}'.format(a[i]), end=' ')
	print("---")

def printing2d(a):
	for i in range(len(a)):
		printing(a[i])
	print("---")

def getMaxOfRow(a):
	ans = abs(a[0])
	for x in a:
		ans = max(ans, abs(x))
	return ans

def swap(a, b):
	tmp = a
	a = b
	b = tmp

# ax = y with a is a square magic
def solve(a, y, partialPivotStrategy = True):					
	if len(a) != len(y):
		raise RuntimeError()
	if not isinstance(a, list):
		a = a.tolist()
	if not isinstance(y, list):
		y = y.tolist()
	n = len(a)
	scaleFactor = list(map(getMaxOfRow, a))
	# add vector y to the end of a
	for i in range(n):
		a[i].append(y[i])
	# go down
	for i in range(n):
		# first element of row is = 0
		if (a[i][i] == 0):
			cs = -1
			for j in range(i+1, n):
				if cs == -1 or a[j][i] < a[cs][i]:
					cs = j
			if cs != -1:
				a[i], a[cs] = a[cs], a[i]
				y[i], y[cs] = y[cs], y[i]
				scaleFactor[i], scaleFactor[cs] = scaleFactor[cs], scaleFactor[i];
		# printing2d(a)
		# partial pivot strategy
		if partialPivotStrategy:
			cs = i
			for j in range(i+1, n):
				if abs(a[j][i]/scaleFactor[j]) > abs(a[cs][i]/scaleFactor[cs]):
					cs = j
			a[i], a[cs] = a[cs], a[i]
			y[i], y[cs] = y[cs], y[i]
			scaleFactor[i], scaleFactor[cs] = scaleFactor[cs], scaleFactor[i];
			# printing2d(a)
		# first element is still no => free variable
		if a[i][i] == 0:
			raise RuntimeError("This may have many solution")
		# divide row
		a[i] = multiply(a[i], 1/a[i][i])
		# eliminate first element of all lower row
		for j in range(i+1, n):
			a[j] = add(a[j], multiply(a[i], -a[j][i]))


	# go up
	for i in range(n-1, -1, -1):
		for j in range(i+1, n):
			a[i] = add(a[i], multiply(a[j], -a[i][j]))
	return [a[i][n] for i in range(0, n)]

def solve_numpy(a, y):
	x = numpy.linalg.solve(a, y)
	return x

def test():
	#a = [[0.00000000001, 1], [1, 1]]
	#y = [1.00000000001, 2]
	a = numpy.random.random((12, 12))
	y = numpy.random.random(12)
	#a = [[0.001, 2, 3], [0, 3, 0], [2, -4, 2]]
	#y = [10, 6, 0]
	printing(solve(copy.deepcopy(a), copy.deepcopy(y)))
	printing(solve_numpy(copy.deepcopy(a), copy.deepcopy(y)))

def main():
	test()

if __name__ == "__main__":
	main()
