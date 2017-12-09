from __future__ import division
import numpy as np
import Regressors
from Features import features

def vector2d(x):
	return x.reshape((x.shape[0], 1))

def split(X, Y, train_size):
	data = np.concatenate((X, Y), axis=1)
	np.random.shuffle(data)
	if train_size < 1:
		split_point = int(train_size*data.shape[0])
		X_train, X_test = data[:split_point,:-1], data[split_point:,:-1]
		Y_train, Y_test = vector2d(data[:split_point,-1]), vector2d(data[split_point:,-1])
	else:
		X_train, X_test = data[:,:-1], data[:,:-1]
		Y_train, Y_test = vector2d(data[:,-1]), vector2d(data[:,-1])
	return X_train, Y_train, X_test, Y_test

data = np.load("DataStandardized.npy")
X_train_valid, Y_train_valid, X_test, Y_test = split(data[:,1:], vector2d(data[:,0]), 0.8)

X_test = np.delete(X_test, [0,1,2,4,6], axis=1)

test_error = 0
iterations = len(features)+1
for iterate in range(iterations):
	average_errors = np.zeros((len(features)+1, 2))
	num_counts = 20
	for count in range(num_counts):
		X_train, Y_train, X_valid, Y_valid = split(X_train_valid, Y_train_valid, 0.75)
		R = Regressors.Polynomial(2)
		R.train(X_train, Y_train)
		errors = [[R.training_error, R.testing_error(X_valid, Y_valid)]]
		if len(features) == 2:
			test_error += R.testing_error(X_test, Y_test)
		for i in range(len(features)):
			X_train_prime = np.delete(X_train, i, axis=1)
			X_valid_prime = np.delete(X_valid, i, axis=1)
			R = Regressors.Polynomial(2)
			R.train(X_train_prime, Y_train)
			errors.append( [R.training_error, R.testing_error(X_valid_prime, Y_valid)] )
		average_errors = average_errors + np.asarray(errors)
	average_errors = average_errors/num_counts
	print "training error: ", average_errors[0,0]
	print "validation error: ", average_errors[0,1]
#	for f in range(len(features)):
#		print features[f] + ": ", average_errors[f,:]
	if iterate < iterations-1:
		f_min = np.argmin(np.sum(average_errors[1:,:]**2, axis=1))
		print "\r\nDropping feature \"" + features[f_min] + "\"\r\n"
		features.pop(f_min)
		X_train_valid = np.delete(X_train_valid, f_min, axis=1)
test_error = test_error/num_counts

print "\r\ntesting error (with only two features): ", test_error
