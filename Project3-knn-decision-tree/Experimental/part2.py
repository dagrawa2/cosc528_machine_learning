from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from algos import *
from tools import *

X = np.load("Arrays\\X.npy")
y = np.load("Arrays\\y.npy")

X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.8)
X, y = None, None

accuracies_tr = []
accuracies_va = []
depths = []

num = 5
K_values = range(2,9)+[16,32]
eps_values = [0, 0.001, 0.01, 0.1]

#K = 5
eps = 0.01
for K in K_values:
#for eps in eps_values:
	D_tr = {"accuracy":0, "precision":0, "recall":0, "specificity":0, "F_score":0}
	D_va = {"accuracy":0, "precision":0, "recall":0, "specificity":0, "F_score":0}
	depth = 0
	for count in range(num):
		X_tr, y_tr, X_va, y_va = train_validation_split(X_train, y_train, num_splits=num, set_idx=count)
		DT = decision_tree(K=K, eps=eps, impurity="entropy")
		DT.train(X_tr, y_tr)
		depth += DT.depth()
		y_tr_pred = DT.predict(X_tr)
		y_va_pred = DT.predict(X_va)
		M_tr = metrics(y_tr, y_tr_pred)
		M_va = metrics(y_va, y_va_pred)
		for key in D_tr.keys():
			D_tr[key] += M_tr[key]
			D_va[key] += M_va[key]
	for key in D_tr.keys():
		D_tr[key] = D_tr[key]/num
		D_va[key] = D_va[key]/num
	accuracies_tr.append(D_tr["accuracy"])
	accuracies_va.append(D_va["accuracy"])
	depths.append(depth/num)

for i in range(len(K_values)):
#for i in range(len(eps_values)):
	print "K: ", K_values[i]
#	print "eps: ", eps_values[i]
	print "depth: ", depths[i]
	print "accuracy_tr: ", accuracies_tr[i]
	print "accuracy_va: ", accuracies_va[i]
	print "\n"

import sys
sys.exit()

fig1 = plt.figure()
sub = fig1.add_subplot(1,1,1)
sub.plot(K_values, accuracies_va)
sub.set_title("Average Accuracy over Validation Sets")
sub.set_xlabel("$K$")
sub.set_ylabel("Accuracy")
fig1.savefig("Plots\\part2", bbox_inches='tight')
#plt.show()

K = 5
eps = 0.01
DT = decision_tree(K=K, eps=eps, impurity="entropy")
DT.train(X_train, y_train)
y_pred = DT.predict(X_test)
M = metrics(y_test, y_pred)

file = open("Tables\\part2.txt", "w")
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
