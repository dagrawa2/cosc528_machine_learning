from __future__ import division
import numpy as np

def metrics(Y_actual, Y_predict):
	TN, FP, FN, TP = 0, 0, 0, 0
	for i in range(len(Y_actual)):
		if Y_actual[i]==0 and Y_predict[i]==0: TN += 1
		elif Y_actual[i]==0 and Y_predict[i]==1: FP += 1
		elif Y_actual[i]==1 and Y_predict[i]==0: FN += 1
		elif Y_actual[i]==1 and Y_predict[i]==1: TP += 1
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

def train_test_split(X, Y, train_size=0.8, random_state=None):
	np.random.seed(random_state)
	N_train = int(train_size*len(Y))
	indeces = np.array(range(len(Y)))
	np.random.shuffle(indeces)
	X_train = X[indeces[:N_train],:]
	Y_train = Y[indeces[:N_train],:]
	X_test = X[indeces[N_train:],:]
	Y_test = Y[indeces[N_train:],:]
	return X_train, Y_train, X_test, Y_test
