from __future__ import division
import numpy as np
from copy import deepcopy
from activations import *
from outputs import *

class input:

	def __init__(self):
		self.Z = None

	def predict(self, X):
		return X

	def forward(self, X):
		self.Z = deepcopy(X)
		return X


class layer:

	def __init__(self, dim_in, dim_out, activation):
		self.activation = activation()
		self.W = np.random.uniform(-0.1, 0.1, (dim_in+1, dim_out))
		self.Z = None

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		return self.activation(X.dot(self.W))

	def forward(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		self.Z = self.activation(X.dot(self.W))
		return self.Z

	def update(self, chain, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		delta = chain*self.activation.gradient(self.Z)
		gradient_W = X.T.dot(delta)/delta.shape[0]
		chain = delta.dot(self.W[:-1,:].T)
		return chain, gradient_W


class dropout:

	def __init__(self, dim_in, dim_out, activation, p=1):
		self.activation = activation()
		self.p = p
		self.W = np.random.uniform(-0.1, 0.1, (dim_in+1, dim_out))
		self.Z = None
		self.mask = None

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		return self.p*self.activation(X.dot(self.W))

	def forward(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		self.Z = self.activation(X.dot(self.W))*np.random.binomial(1, self.p, (1, self.W.shape[1]))
		return self.Z

	def update(self, chain, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		delta = chain*self.activation.gradient(self.Z)
		gradient_W = X.T.dot(delta)/delta.shape[0]
		chain = delta.dot(self.W[:-1,:].T)
		return chain, gradient_W


class output:

	def __init__(self, dim_in, dim_out, activation):
		self.activation = activation()
		self.W = np.random.uniform(-0.1, 0.1, (dim_in+1, dim_out))
		self.Z = None

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		return self.activation(X.dot(self.W))

	def forward(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		self.Z = self.activation(X.dot(self.W))
		return self.Z

	def init_chain(self, Y):
		return self.Z-Y

	def update(self, chain, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		delta = chain
		gradient_W = X.T.dot(delta)/delta.shape[0]
		chain = delta.dot(self.W[:-1,:].T)
		return chain, gradient_W
