from __future__ import division
import numpy as np
from algos2 import *
from tools import *

X = np.load("Arrays\\X.npy")
Y = np.load("Arrays\\Y.npy")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, train_size=0.75, random_state=123)
X, Y = None, None

mu = np.mean(X_train, axis=0, keepdims=True)
sigma = np.std(X_train, axis=0, keepdims=True)

X_train = (X_train-mu)/sigma
X_test = (X_test-mu)/sigma

pc = 57
U, s, V = np.linalg.svd(X_train)
W = V[:,:pc]
U, s, V = None, None, None

X_train = X_train.dot(W)
X_test = X_test.dot(W)


model = NN( \
[pc,11,1], \
activations=[logistic, logistic], \
learning_rate = 0.01, \
mini_batch_size = 1, \
random_state=2718 \
)

#for l in range(len(model.W)):
#	model.W[l] = np.load("Arrays\\W_"+str(l)+".npy")
#	model.b[l] = np.load("Arrays\\b_"+str(l)+".npy")

model.train(X_train, Y_train, test=(X_test,Y_test), epochs=100, monitor_freq=1)
print "\n---\n"
model.eta = 0.0001
model.train(X_train, Y_train, test=(X_test,Y_test), epochs=21, monitor_freq=1)

#for l in range(len(model.W)):
#	np.save("Arrays\\W_"+str(l)+"_pca.npy", model.W[l])
#	np.save("Arrays\\b_"+str(l)+"_pca.npy", model.b[l])
