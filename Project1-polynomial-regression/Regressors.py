from __future__ import division
import numpy as np
import Terms

class Linear():

	def __init__(self):
		self.params = None
		self.training_error = None

	def augment(self, X):
		return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

	def train(self, X, Y):
		X = self.augment(X)
		self.params = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
		self.training_error = np.sum((np.dot(X, self.params)-Y)**2)/X.shape[0]

	def predict(self, X):
		return np.dot(self.augment(X), self.params)

	def testing_error(self, X, Y):
		return np.sum((self.predict(X)-Y)**2)/X.shape[0]


class Polynomial():

	def __init__(self, order):
		self.order = order
		self.params = None
		self.training_error = None

	def augment(self, X):
		num_vars = X.shape[1]
		X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
		for order in range(2, self.order+1):
			terms = Terms.GenerateTerms(num_vars, order)
			for term in terms:
				prod = np.ones((X.shape[0]))
				for i in term:
					prod = prod*X[:,i]
				prod = prod.reshape((X.shape[0], 1))
				X = np.concatenate((X, prod), axis=1)
		return X

	def train(self, X, Y):
		X = self.augment(X)
		self.params = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
		self.training_error = np.sum((np.dot(X, self.params)-Y)**2)/X.shape[0]

	def predict(self, X):
		return np.dot(self.augment(X), self.params)

	def testing_error(self, X, Y):
		return np.sum((self.predict(X)-Y)**2)/X.shape[0]


#N = 100
#x = np.random.uniform(0, 1, (N, 1))
#y = np.random.uniform(0, 1, (N, 1))
#z = 1+2*x+3*y + np.random.normal(0, 0.25, (N, 1))
#
#X = np.concatenate((x, y), axis=1)
#L = Linear()
#L.train(X, z)
#print "X.shape = ", X.shape
#print "z.shape = ", z.shape
#print "params = ", L.params
#print "training error = ", L.training_error