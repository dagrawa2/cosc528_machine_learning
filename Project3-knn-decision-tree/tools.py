from __future__ import division
import numpy as np

def metrics(y_actual, y_predict):
	TN, FP, FN, TP = 0, 0, 0, 0
	for i in range(len(y_actual)):
		if y_actual[i]==0 and y_predict[i]==0: TN += 1
		elif y_actual[i]==0 and y_predict[i]==1: FP += 1
		elif y_actual[i]==1 and y_predict[i]==0: FN += 1
		elif y_actual[i]==1 and y_predict[i]==1: TP += 1
	TPR = TP/(TP+FN)
	PPV = TP/(TP+FP)
	D = { \
	"confusion_matrix": np.array([[TN, FP], [FN, TP]]), \
	"accuracy": (TN+TP)/(TN+FP+FN+TP), \
	"recall": TPR, \
	"precision": PPV, \
	"specificity": TN/(TN+FP), \
	"F_score": PPV*TPR/(PPV+TPR) \
	}
	return D

def train_test_split(X, y, train_size=0.8):
	N_train = int(train_size*len(y))
	indeces = np.array(range(len(y)))
	np.random.shuffle(indeces)
	X_train = X[indeces[:N_train]]
	y_train = y[indeces[:N_train]]
	X_test = X[indeces[N_train:]]
	y_test = y[indeces[N_train:]]
	return X_train, y_train, X_test, y_test

def train_validation_split(X_train, y_train, num_splits=5, set_idx=0):
	train_size = X_train.shape[0]
	valid_size = int(train_size/num_splits)
	remainder = train_size%num_splits
	valid_sizes = [valid_size]*num_splits
	for i in range(remainder):
		valid_sizes[i] += 1
	split_idx = [None] + [sum(valid_sizes[:i]) for i in range(1, num_splits)] + [None]
	all_indeces = np.arange(train_size)
	indeces = all_indeces[split_idx[set_idx]: split_idx[set_idx+1]]
	mask = np.ones((train_size), bool)
	mask[indeces] = False
	X_tr = X_train[mask, :]
	y_tr = y_train[mask]
	X_va = X_train[indeces, :]
	y_va = y_train[indeces]
	return X_tr, y_tr, X_va, y_va
