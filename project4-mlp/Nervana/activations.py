from __future__ import division
import numpy as np

class logistic:

	def __init__(self):
		pass

	def __call__(self, X):
		return 1/(1+np.exp(-X))

	def gradient(self, Z):
		return Z*(1-Z)


class relu:

	def __init__(self):
		pass

	def __call__(self, X):
		return np.maximum(0, X)

	def gradient(self, Z):
		return np.heaviside(Z, 0)


class tanh:

	def __init__(self):
		pass

	def __call__(self, X):
		return np.tanh(X)

	def gradient(self, Z):
		return 1-Z**2
