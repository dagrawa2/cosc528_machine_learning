from __future__ import division
import numpy as np
import time
from copy import deepcopy
from activations import *
from layers import *
import outputs

class NN:

	def __init__(self, layers, learning_rate=0.01, mini_batch_size=1, random_state=None):
		np.random.seed(random_state)
		self.layers = [input()] + layers
		self.eta = learning_rate
		self.mbs = mini_batch_size
		self.loss = layers[-1].activation.loss
		self.debug = False

	def predict(self, X):
		Z = deepcopy(X)
		for layer in self.layers:
			Z = layer.predict(Z)
		return Z

	def forward(self, X):
		Z = deepcopy(X)
		for layer in self.layers:
			Z = layer.forward(Z)
		return Z

	def backward(self, Y):
		chain = self.layers[-1].init_chain(Y)
		for layer, prev_layer in zip(self.layers[1:], self.layers[:-1])[::-1]:
			chain, gradient_W = layer.update(chain, prev_layer.Z)
			if self.debug:
				h = 10**-5
				delta_W = np.zeros_like(layer.W)
				i = np.random.randint(0, layer.W.shape[0])
				j = np.random.randint(0, layer.W.shape[1])
				delta_W[i,j] = 1
				W_0 = deepcopy(layer.W)
				layer.W += h*delta_W
				L_1 = self.loss(self.Y_temp, self.predict(self.X_temp))
				layer.W -= 2*h*delta_W
				L_2 = self.loss(self.Y_temp, self.predict(self.X_temp))
				print (L_1-L_2)/(2*h) - gradient_W[i,j]
				layer.W = W_0
				self.debug = False
			layer.W -= self.eta*gradient_W

	def train(self, X_train, Y_train, epochs=1, test=None, monitor_freq=9999):
		output_layer = self.layers[-1].activation.__class__
		indices = np.arange(X_train.shape[0])
		time_0 = time.time()
		for epoch in range(epochs):
			if epoch%monitor_freq == 0:
				if output_layer == linear:
					error_str = self.error_string(X_train, Y_train, test=test)
					print str(epoch)+", "+error_str+","+str(np.round(time.time()-time_0, 3))
				else:
					accuracy_str = self.accuracy_string(X_train, Y_train, test=test)
					print str(epoch)+","+accuracy_str+","+str(np.round(time.time()-time_0, 3))
			np.random.shuffle(indices)
			X = X_train[indices,:]
			Y = Y_train[indices,:]
			count = 0
			for X_batch,Y_batch in [(X[i:i+self.mbs,:], Y[i:i+self.mbs,:]) for i in range(0,X.shape[0],self.mbs)]:
				count += 1
				if epoch > 50 and count%int(0.1*X.shape[0]/self.mbs) == 0:
					self.debug = True
					self.X_temp = X_batch
					self.Y_temp = Y_batch
				self.forward(X_batch)
				self.backward(Y_batch)

	def accuracy_string(self, X_train, Y_train, test=None):
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

	def error_string(self, X_train, Y_train, test=None):
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
