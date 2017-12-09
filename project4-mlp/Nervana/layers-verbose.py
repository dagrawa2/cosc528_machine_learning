from __future__ import division
import numpy as np
from activations import *

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


class linear_output:

	def __init__(self, dim_in, dim_out):
		self.name = "linear"
		self.W = np.random.uniform(-0.1, 0.1, (dim_in+1, dim_out))
		self.Z = None

	def loss(self, Y, Z):
		return np.mean(np.sum((Z-Y)**2, axis=1))

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		return X.dot(self.W)

	def forward(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		self.Z = X.dot(self.W)
		return self.Z

	def init_chain(self, Y):
		return self.Z-Y

	def update(self, chain, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		delta = chain
		gradient_W = X.T.dot(delta)/delta.shape[0]
		chain = delta.dot(self.W[:-1,:].T)
		return chain, gradient_W


class logistic_output:

	def __init__(self, dim_in, dim_out):
		self.name = "logistic"
		self.W = np.random.uniform(-0.1, 0.1, (dim_in+1, dim_out))
		self.Z = None

	def loss(self, Y, Z):
		return -np.mean(Y*np.log(Z)+(1-Y)*np.log(1-Z))

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		return 1/(1+np.exp(-X.dot(self.W)))

	def forward(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		self.Z = 1/(1+np.exp(-X.dot(self.W)))
		return self.Z

	def init_chain(self, Y):
		return self.Z-Y

	def update(self, chain, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		delta = chain
		gradient_W = X.T.dot(delta)/delta.shape[0]
		chain = delta.dot(self.W[:-1,:].T)
		return chain, gradient_W


class softmax_output:

	def __init__(self, dim_in, dim_out):
		self.name = "softmax"
		self.W = np.random.uniform(-0.1, 0.1, (dim_in+1, dim_out))
		self.Z = None

	def loss(self, Y, Z):
		return -np.mean(np.sum(Y*np.log(Z), axis=1))

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		temp = np.exp(X.dot(self.W))
		return temp/np.sum(temp, axis=1, keepdims=True)

	def forward(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		temp = np.exp(X.dot(self.W))
		self.Z = temp/np.sum(temp, axis=1, keepdims=True)
		return self.Z

	def init_chain(self, Y):
		return self.Z-Y

	def update(self, chain, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		delta = chain
		gradient_W = X.T.dot(delta)/delta.shape[0]
		chain = delta.dot(self.W[:-1,:].T)
		return chain, gradient_W
