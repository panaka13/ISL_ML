import numpy
import matplotlib.pyplot

# A^T*A*x = A^T*y
def project(A, y):
	trans = numpy.transpose(A)
	newA = numpy.dot(trans, A)
	newY = numpy.dot(trans, y)
	import LinearSystem
	return LinearSystem.solve(newA, newY)

def degreePolynomial(x, y, degree):
	newX = []
	for i in range(len(x)):
		arr = []
		tmp = 1
		for j in range(degree+1):
			arr.append(tmp)
			tmp *= x[i]
		arr.reverse()
		newX.append(arr)
	return project(newX, y)

def graphProjection(x, y, function, label=""):
	import matplotlib
	matplotlib.pyplot.plot(x, y, color="green")
	newX = [x[0]]
	for i in range(1, len(x)):
		while newX[-1] < x[i]:
			newX.append(newX[-1]+0.1)
	matplotlib.pyplot.plot(x, list(map(function, x)), color="red")
	matplotlib.pyplot.title(label)
	matplotlib.pyplot.show()
