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


model = NN( \
[57,11,1], \
activations=[logistic, logistic], \
learning_rate = 0.01, \
mini_batch_size = 1, \
random_state=2718 \
)

for l in range(len(model.W)):
	model.W[l] = np.load("Arrays\\W_"+str(l)+".npy")
	model.b[l] = np.load("Arrays\\b_"+str(l)+".npy")

Y_pred = model.predict(X_test)
Y_pred[Y_pred>0.5] = 1
Y_pred[Y_pred<=0.5] = 0

y_test = Y_test.reshape((Y_test.shape[0]))
y_pred = Y_pred.reshape((Y_pred.shape[0]))

M = metrics(y_test, y_pred)

file = open("Tables\\table3.txt", "w")

file.write("% confusion matrix\n")
C = M["confusion_matrix"]
file.write("\\begin{bmatrix}\n")
file.write(str(C[0,0])+" & "+str(C[0,1])+" \\\\\n")
file.write(str(C[1,0])+" & "+str(C[1,1])+"\n")
file.write("\\end{bmatrix}\n\n")
file.write("% metrics on test set\n")
file.write("\\begin{tabular}{|c|c|} \\hline\n")
file.write("Accuracy & "+str(np.round(M["accuracy"], 3))+" \\\\ \\hline\n")
file.write("Recall & "+str(np.round(M["recall"], 3))+" \\\\ \\hline\n")
file.write("Precision & "+str(np.round(M["precision"], 3))+" \\\\ \\hline\n")
file.write("Specificity & "+str(np.round(M["specificity"], 3))+" \\\\ \\hline\n")
file.write("F-score & "+str(np.round(M["F_score"], 3))+" \\\\ \\hline\n")
file.write("\\end{tabular}\n")

file.close()
