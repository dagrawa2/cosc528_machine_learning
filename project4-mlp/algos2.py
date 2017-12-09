from __future__ import division
import numpy as np
import time
from copy import deepcopy

def linear(X):
	return X

def logistic(X):
	return 1/(1+np.exp(-X))

def logistic_deriv(Y):
	return Y*(1-Y)

def relu(X):
	return np.maximum(0, X)

def relu_deriv(Y):
		return np.heaviside(Y, 0)

def softmax(X):
	temp = np.exp(X)
	return temp/np.sum(temp, axis=1, keepdims=True)

def cross_entropy(Y, Z):
	return -np.mean(Y*np.where(Y>0, np.log(Z), Z)+(1-Y)*np.where(Y<1, np.log(1-Z), Z))

def entropy(Y, Z):
	return -np.mean(np.sum(Y*np.where(Y>0, np.log(Z), Z), axis=1))

def square_error(Y, Z):
	return np.mean(np.sum((Z-Y)**2, axis=1)/2)


class NN:

	def __init__(self, layers, activations, learning_rate=0.1, mini_batch_size=1, random_state=None):
		self.layers = layers
		self.activations = activations
		self.activation_derivs = []
		for act in activations[:-1]:
			if act == logistic: self.activation_derivs.append(logistic_deriv)
			elif act == relu: self.activation_derivs.append(relu_deriv)
		self.activation_derivs.append(None)
		if self.activations[-1] == logistic: self.loss = cross_entropy
		elif self.activations[-1] == softmax: self.loss = entropy
		elif self.activations[-1] == linear: self.loss = square_error
		self.eta = learning_rate
		self.mbs = mini_batch_size
		np.random.seed(random_state)
		self.W = [np.random.uniform(-0.01, 0.01, (m, n)) for m,n in zip(layers[:-1], layers[1:])]
		self.b = [np.random.uniform(-0.01, 0.01, (1, n)) for n in layers[1:]]

	def predict(self, X):
		A = np.atleast_2d(deepcopy(X))
		for act,W,b in zip(self.activations, self.W, self.b):
			A = act(A.dot(W)+b)
		return A

	def forward(self, X):
		a = [np.atleast_2d(deepcopy(X))]
		for act,W,b in zip(self.activations, self.W, self.b):
			a.append( act(a[-1].dot(W)+b) )
		return a

	def backward(self, Y, a):
		deltas = [a[-1]-Y]
		for W,act_deriv,a_ in zip(self.W[1:], self.activation_derivs[:-1], a[1:-1])[::-1]:
			deltas.append( deltas[-1].dot(W.T)*act_deriv(a_) )
#			deltas.append( deltas[-1].dot(W.T)*np.sum(act_deriv(a_), keepdims=True) )
		deltas.reverse()
		for W,b,a_,delta in zip(self.W, self.b, a[:-1], deltas):
			W -= self.eta/Y.shape[0]*a_.T.dot(delta)
			b -= self.eta/Y.shape[0]*np.sum(delta, axis=0, keepdims=True)

	def train(self, X_train, Y_train, test=None, epochs=1, monitor_freq=9999):
		print "epoch,loss,trainingScore,validationScore,time"
		indices = np.arange(X_train.shape[0])
		time_0 = time.time()
		for epoch in range(epochs):
			if epoch%monitor_freq == 0:
				if self.loss == cross_entropy: performance_str = self.logistic_performance(X_train, Y_train, test=test)
				elif self.loss == entropy: performance_str = self.softmax_performance(X_train, Y_train, test=test)
#				elif self.loss == square_error: performance_str = self.linear_performance(X_train, Y_train, test=test)
				elif self.loss == square_error: performance_str = self.logistic_performance(X_train, Y_train, test=test)
				print str(epoch)+","+performance_str+","+str(np.round(time.time()-time_0, 3))
			np.random.shuffle(indices)
			X = X_train[indices,:]
			Y = Y_train[indices,:]
			for X_batch,Y_batch in [(X[i:i+self.mbs,:], Y[i:i+self.mbs,:]) for i in range(0,X.shape[0],self.mbs)]:
				self.backward(Y_batch, self.forward(X_batch))

	def logistic_performance(self, X_train, Y_train, test=None):
		Y_pred = self.predict(X_train)
		L = str(np.round(self.loss(Y_train, Y_pred), 3))
		Y_pred[Y_pred>0.5] = 1
		Y_pred[Y_pred<=0.5] = 0
		accuracy = 0
		for i in range(len(Y_train)):
			if Y_train[i] == Y_pred[i]: accuracy += 1
		accuracy = str(np.round(accuracy/Y_train.shape[0], 3))
		if test:
			X_test = test[0]
			Y_test = test[1]
			Y_pred = self.predict(X_test)
			Y_pred[Y_pred>0.5] = 1
			Y_pred[Y_pred<=0.5] = 0
			accuracy_test = 0
			for i in range(len(Y_test)):
				if Y_test[i] == Y_pred[i]: accuracy_test += 1
			accuracy_test = str(np.round(accuracy_test/Y_test.shape[0], 3))
			accuracy = accuracy+","+accuracy_test
		return L+","+accuracy

	def softmax_performance(self, X_train, Y_train, test=None):
		Y_pred = self.predict(X_train)
		L = str(np.round(self.loss(Y_train, Y_pred), 3))
		Y_train = np.argmax(Y_train, axis=1)
		Y_pred = np.argmax(Y_pred, axis=1)
		accuracy = 0
		for i in range(len(Y_train)):
			if Y_train[i] == Y_pred[i]: accuracy += 1
		accuracy = str(np.round(accuracy/Y_train.shape[0], 3))
		if test:
			X_test = test[0]
			Y_test = test[1]
			Y_pred = self.predict(X_test)
			Y_test = np.argmax(Y_test, axis=1)
			Y_pred = np.argmax(Y_pred, axis=1)
			accuracy_test = 0
			for i in range(len(Y_test)):
				if Y_test[i] == Y_pred[i]: accuracy_test += 1
			accuracy_test = str(np.round(accuracy_test/Y_test.shape[0], 3))
			accuracy = accuracy+","+accuracy_test
		return L+","+accuracy

	def linear_performance(self, X_train, Y_train, test=None):
		Y_pred = self.predict(X_train)
		L = str(np.round(self.loss(Y_train, Y_pred), 3))
		error = str(np.round( np.mean(np.sqrt(np.sum((Y_pred-Y_train)**2, axis=1))), 3))
		if test:
			X_test = test[0]
			Y_test = test[1]
			Y_pred = self.predict(X_test)
			error_test = str(np.round( np.mean(np.sqrt(np.sum((Y_pred-Y_test)**2, axis=1))), 3))
			error = error+","+error_test
		return L+","+error
