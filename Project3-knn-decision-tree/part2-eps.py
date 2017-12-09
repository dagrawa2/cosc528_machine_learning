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
depths = []

num_splits = 20
K_values = range(2,9)+[16,32]
eps_values = [0, 0.001, 0.01, 0.1]

K = 5
#eps = 0.01
#for K in K_values:
for eps in eps_values:
	np.random.seed(314159)
	a_tr = []
	a_va = []
	depth = []
	for count in range(num_splits):
		X_tr, y_tr, X_va, y_va = train_test_split(X_train, y_train, train_size=0.8)
		DT = decision_tree(K=K, eps=eps, impurity="entropy")
		DT.train(X_tr, y_tr)
		y_tr_pred = DT.predict(X_tr)
		y_va_pred = DT.predict(X_va)
		M_tr = metrics(y_tr, y_tr_pred)
		M_va = metrics(y_va, y_va_pred)
		a_tr.append(M_tr["accuracy"])
		a_va.append(M_va["accuracy"])
		depth.append(DT.depth())
	a_tr = np.array(a_tr)
	a_va = np.array(a_va)
	depth = np.array(depth)
	accuracies_tr.append(np.mean(a_tr))
	accuracies_va.append(np.mean(a_va))
	accuracies_va_std.append(np.std(a_va))
	depths.append(np.mean(depth))

file = open("Outs\\part2-eps-entropy.txt", "w")
#for i in range(len(K_values)):
for i in range(len(eps_values)):
#	file.write("K: "+str(K_values[i])+"\r\n")
	file.write("eps: "+str(eps_values[i])+"\r\n")
	file.write("depth: "+str(depths[i])+"\r\n")
	file.write("accuracy_tr: "+str(accuracies_tr[i])+"\r\n")
	file.write("accuracy_va: "+str(accuracies_va[i])+"\r\n")
	file.write("accuracy_va_std: "+str(accuracies_va_std[i])+"\r\n\r\n")
file.close()

#ints = np.arange(1, len(K_values)+1)
ints = np.arange(1, len(eps_values)+1)
accuracies_va = np.array(accuracies_va)
accuracies_va_std = np.array(accuracies_va_std)
depths = np.array(depths)

fig1 = plt.figure()
sub = fig1.add_subplot(1,1,1)
sub.plot(ints, accuracies_va)
sub.plot(ints, accuracies_va-accuracies_va_std, linestyle="--")
sub.plot(ints, accuracies_va+accuracies_va_std, linestyle="--")
sub.set_xticks(ints)
sub.set_xticklabels(eps_values)
sub.set_title("Accuracies over Validation Sets")
sub.set_xlabel("$\epsilon$")
sub.set_ylabel("Accuracy")
fig1.savefig("Plots\\part2-eps-entropy.png", bbox_inches='tight')
#plt.show()

# remove for K
import sys
sys.exit()

fig2 = plt.figure()
sub = fig2.add_subplot(1,1,1)
sub.plot(ints, depths)
sub.set_xticks(ints)
sub.set_xticklabels(K_values)
sub.set_title("Average Realized Depths")
sub.set_xlabel("$K$")
sub.set_ylabel("Depth")
fig2.savefig("Plots\\part2-K-entropy-depth.png", bbox_inches='tight')
#plt.show()
