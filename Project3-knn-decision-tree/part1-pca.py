from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from algos import *
from tools import *

X_train = np.load("Arrays\\X_train.npy")
y_train = np.load("Arrays\\y_train.npy")
X_test = np.load("Arrays\\X_test.npy")
y_test = np.load("Arrays\\y_test.npy")

accuracies_tr = []
accuracies_va = []
accuracies_va_std = []

num_splits = 20
K_values = range(2,9)+[16,32]
var_suff = 0.9

avg_dim = 0

for K in K_values:
	np.random.seed(314159)
	a_tr = []
	a_va = []
	for count in range(num_splits):
		X_tr, y_tr, X_va, y_va = train_test_split(X_train, y_train, train_size=0.8)
		mean_tr = np.mean(X_tr, axis=0, keepdims=True)
		U, s, V = np.linalg.svd(X_tr-mean_tr)
		U = None
		var_total = np.sum(s**2)
		var_accum = 0
		dim = 0
		for k in range(len(s)):
			var_accum += s[k]**2
			dim += 1
			if var_accum/var_total >= var_suff: break
		W = V[:,:dim]
		avg_dim += dim
		V = None
		X_tr = (X_tr-mean_tr).dot(W)
		X_va = (X_va-mean_tr).dot(W)
		KNN = knn(K)
		KNN.train(X_tr, y_tr)
		y_tr_pred = KNN.predict(X_tr)
		y_va_pred = KNN.predict(X_va)
		M_tr = metrics(y_tr, y_tr_pred)
		M_va = metrics(y_va, y_va_pred)
		a_tr.append(M_tr["accuracy"])
		a_va.append(M_va["accuracy"])
	a_tr = np.array(a_tr)
	a_va = np.array(a_va)
	accuracies_tr.append(np.mean(a_tr))
	accuracies_va.append(np.mean(a_va))
	accuracies_va_std.append(np.std(a_va))

avg_dim = avg_dim/(num_splits*len(K_values))

file = open("Outs\\part1-pca.txt", "w")
file.write("average PCA dimension: "+str(avg_dim)+"\r\n\r\n")
for i in range(len(K_values)):
	file.write("K: "+str(K_values[i])+"\r\n")
	file.write("accuracy_tr: "+str(accuracies_tr[i])+"\r\n")
	file.write("accuracy_va: "+str(accuracies_va[i])+"\r\n")
	file.write("accuracy_va_std: "+str(accuracies_va_std[i])+"\r\n\r\n")
file.close()

ints = np.arange(1, len(K_values)+1)
accuracies_va = np.array(accuracies_va)
accuracies_va_std = np.array(accuracies_va_std)
fig1 = plt.figure()
sub = fig1.add_subplot(1,1,1)
sub.plot(ints, accuracies_va)
sub.plot(ints, accuracies_va-accuracies_va_std, linestyle="--")
sub.plot(ints, accuracies_va+accuracies_va_std, linestyle="--")
sub.set_xticks(ints)
sub.set_xticklabels(K_values)
sub.set_title("Accuracies over Validation Sets")
sub.set_xlabel("$K$")
sub.set_ylabel("Accuracy")
fig1.savefig("Plots\\part1-pca", bbox_inches='tight')
#plt.show()
