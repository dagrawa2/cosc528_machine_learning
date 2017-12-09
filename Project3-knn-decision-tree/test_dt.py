from __future__ import division
import numpy as np
from algos import *

X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([0,0,0,1,0,1,0,1])

D = decision_tree(K=3, eps=0.001)
D.train(X, y)
y_pred = D.predict(X)

print "actual predicted"
for i in range(len(y)):
	print y[i], y_pred[i]
