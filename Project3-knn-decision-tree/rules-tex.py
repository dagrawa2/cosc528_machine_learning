from __future__ import division
import networkx as nx
import numpy as np
from algos import *
from tools import *

X_train = np.load("Arrays\\X_train.npy")
y_train = np.load("Arrays\\y_train.npy")
X_test = np.load("Arrays\\X_test.npy")
y_test = np.load("Arrays\\y_test.npy")

#features = ["F_"+str(i) for i in range(9)]
features = [ \
"ClumpThickness", \
"UniformityOfCellSize", \
"UniformityOfCellShape", \
"MarginalAdhesion", \
"SingleEpithelialCellSize", \
"BareNuclei", \
"BlandChromatin", \
"NormalNucleoli", \
"Mitoses" \
]

np.random.seed(314159)

K = 5
eps = 0.01
DT = decision_tree(K=K, eps=eps, impurity="entropy")
DT.train(X_train, y_train)
#y_pred = DT.predict(X_test)
#M = metrics(y_test, y_pred)

leaves = [N for N in DT.nodes() if DT.out_degree(N) == 0]
paths = []
for N in leaves:
	path = [N] + list(nx.ancestors(DT, N))
	path.sort()
	paths.append(path)

file = open("rules-tex-out.txt", "w")

count = 1
for path in paths:
	file.write("\\textbf{Path "+str(count)+":} \\\\\n")
	file.write("Conditions:\n")
	file.write("\\begin{itemize}\n")
	count += 1
	for i in range(len(path[:-1])):
		N = path[i]
		O = path[i+1]
		symbol = DT.edge[N][O]["branch"]
		if symbol == "less":
			file.write("\\item $\\mbox{"+features[DT.node[N]["feature"]]+"} \leq "+str(np.round(DT.node[N]["value"], 3))+"$\n")
		elif symbol == "greater":
			file.write("\\item \\mbox{"+features[DT.node[N]["feature"]]+"} > "+str(np.round(DT.node[N]["value"], 3))+"\n")
	N = path[-1]
	file.write("\\end{itemize}\n")
	file.write("Prediction: $P(\\mbox{class} 1) = "+str(np.round(DT.node[N]["p"], 3))+"$\n\n")
	file.write("\\[\\quad\\]\n\n\\[\\quad\\]\n\n")

file.close()