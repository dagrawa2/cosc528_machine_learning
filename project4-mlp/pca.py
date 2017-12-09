from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from tools import *

X = np.load("Arrays\\X.npy")
Y = np.load("Arrays\\Y.npy")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, train_size=0.75, random_state=123)
X, Y = None, None

mu = np.mean(X_train, axis=0, keepdims=True)
sigma = np.std(X_train, axis=0, keepdims=True)

X_train = (X_train-mu)/sigma
X_test = (X_test-mu)/sigma

U, s, V = np.linalg.svd(X_train)
U, V = None, None

eigs = s**2
var_tot = np.sum(eigs)

vars = []
sum = 0
print "numPC,fractionOfVariance"
for i in range(len(eigs)):
	sum += eigs[i]
	vars.append(sum/var_tot)
	print str(i+1)+","+str(np.round(sum/var_tot, 3))

vars = np.array(vars)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(1+np.arange(len(vars)), vars)
ax.set_title("Fraction of Total Variance Covered by Principal Components")
ax.set_xlabel("Num. of PCs $d$")
ax.set_ylabel("Fraction of Variance")
fig.savefig("Plots\\pca.png", bbox_inches='tight')
#plt.show()
