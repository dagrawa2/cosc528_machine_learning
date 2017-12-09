from __future__ import division
import numpy as np

class linear:

	def __init__(self):
		pass

	def __call__(self, X):
		return X

	def loss(self, Y, Z):
		return np.mean(np.sum((Z-Y)**2, axis=1))


class logistic:

	def __init__(self):
		pass

	def __call__(self, X):
		return 1/(1+np.exp(-X))

	def loss(self, Y, Z):
		return -np.mean(Y*np.where(Y>0, np.log(Z), Z)+(1-Y)*np.where(Y<1, np.log(1-Z), 1-Z))


class softmax:

	def __init__(self):
		pass

	def __call__(self, X):
		temp = np.exp(X)
		return temp/np.sum(temp, axis=1, keepdims=True)

	def loss(self, Y, Z):
		return -np.mean(np.sum(Y*np.log(Z), axis=1))
