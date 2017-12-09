# this file uses Data.npy instead of DataStandardized.npy
# the mean and std of only the training set is used for standardization

from __future__ import division
import numpy as np
import Regressors

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

data = np.load("Data.npy")
X_train_valid, Y_train_valid, X_test, Y_test = split(data[:,1:], vector2d(data[:,0]), 0.8)

num_counts = 10
max_degree = 4
average_errors = np.zeros((max_degree, 2))
test_error = 0
for count in range(num_counts):
	X_train, Y_train, X_valid, Y_valid = split(X_train_valid, Y_train_valid, 0.75)
	x_mean, x_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
	y_mean, y_std = np.mean(Y_train), np.std(Y_train)
	X_train = (X_train-x_mean)/x_std
	Y_train = (Y_train-y_mean)/y_std
	X_valid = (X_valid-x_mean)/x_std
	Y_valid = (Y_valid-y_mean)/y_std
	errors = []
	for d in range(1, max_degree+1):
		R = Regressors.Polynomial(d)
		R.train(X_train, Y_train)
		errors.append( [R.training_error, R.testing_error(X_valid, Y_valid)] )
		if d == 2:
			test_error += R.testing_error((X_test-x_mean)/x_std, (Y_test-y_mean)/y_std)
	average_errors = average_errors + np.asarray(errors)
average_errors = average_errors/num_counts
test_error = test_error/num_counts

for d in range(max_degree):
	print "Degree = ", d+1, ":"
	print "Training error: ", average_errors[d, 0]
	print "Validation error: ", average_errors[d, 1]
	print "\n"

print "Degree = 2:"
print "Testing error: ", test_error