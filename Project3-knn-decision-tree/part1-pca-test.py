from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from algos import *
from tools import *

X_train = np.load("Arrays\\X_train.npy")
y_train = np.load("Arrays\\y_train.npy")
X_test = np.load("Arrays\\X_test.npy")
y_test = np.load("Arrays\\y_test.npy")

var_suff = 0.9

np.random.seed(314159)

mean_train = np.mean(X_train, axis=0, keepdims=True)
U, s, V = np.linalg.svd(X_train-mean_train)
U = None
var_total = np.sum(s**2)
var_accum = 0
dim = 0
for k in range(len(s)):
	var_accum += s[k]**2
	dim += 1
	if var_accum/var_total >= var_suff: break
W = V[:,:dim]
V = None
X_train = (X_train-mean_train).dot(W)
X_test = (X_test-mean_train).dot(W)

K = 4
KNN = knn(K)
KNN.train(X_train, y_train)
y_pred = KNN.predict(X_test)
M = metrics(y_test, y_pred)

file = open("Tables\\part1-pca.txt", "w")
file.write("% PCA dimension: "+str(dim)+" \r\n\r\n")
file.write("% confusion matrix\r\n")
C = M["confusion_matrix"]
file.write("\\begin{bmatrix}\r\n")
file.write(str(C[0,0])+" & "+str(C[0,1])+" \\\\\r\n")
file.write(str(C[1,0])+" & "+str(C[1,1])+"\r\n")
file.write("\\end{bmatrix}\r\n\r\n")
file.write("% metrics on test set\r\n")
file.write("\\begin{tabular}{|c|c|} \\hline\r\n")
file.write("Accuracy & "+str(np.round(M["accuracy"], 3))+" \\\\ \\hline\r\n")
file.write("Recall & "+str(np.round(M["recall"], 3))+" \\\\ \\hline\r\n")
file.write("Precision & "+str(np.round(M["precision"], 3))+" \\\\ \\hline\r\n")
file.write("Specificity & "+str(np.round(M["specificity"], 3))+" \\\\ \\hline\r\n")
file.write("F-score & "+str(np.round(M["F_score"], 3))+" \\\\ \\hline\r\n")
file.write("\\end{tabular}\r\n")
file.close()
