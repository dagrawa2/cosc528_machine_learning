from __future__ import division
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = np.load("Arrays\\X.npy")
Y = np.load("Arrays\\Y.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)
X, Y = None, None

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


grids = [ \
( np.logspace(-3, 3, 7), np.logspace(-3, 3, 7)), \
( np.concatenate((np.linspace(1, 9, 9), np.linspace(10, 100, 10)), axis=0), np.concatenate((np.linspace(0.01, 0.09, 9), np.linspace(0.1, 1, 10)), axis=0)), \
( np.linspace(5, 7, 21), np.linspace(0.09, 0.11, 21)), \
( np.linspace(5.7, 5.9, 21), np.linspace(0.098, 0.1, 21)) \
]


#for num in range(1, len(grids)+1):
for num in [4]:
	print "Grid "+str(num)+" . . . "

	C_range, gamma_range = grids[num-1]

	param_grid = {"C":C_range, "gamma":gamma_range}
	cv = StratifiedShuffleSplit(n_splits=2, train_size=0.8, random_state=456)
	clf = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	clf.fit(X_train, Y_train)

	file = open("Out\\out_"+str(num)+".txt", "w")
	file.write("Best parameters:\n")
	file.write("\tC: "+str(np.round(clf.best_params_["C"], 5))+"\n")
	file.write("\tgamma: "+str(np.round(clf.best_params_["gamma"], 5))+"\n")
	file.write("Best score: "+str(np.round(clf.best_score_, 5)))
	file.close()

	scores = clf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

	np.save("Plot_data\\Plot_"+str(num)+"\\C_range.npy", C_range)
	np.save("Plot_data\\Plot_"+str(num)+"\\gamma_range.npy", gamma_range)
	np.save("Plot_data\\Plot_"+str(num)+"\\scores.npy", scores)


file = open("Out\\out_test.txt", "w")
file.write("Best parameters:\n")
file.write("\tC: "+str(np.round(clf.best_params_["C"], 5))+"\n")
file.write("\tgamma: "+str(np.round(clf.best_params_["gamma"], 5))+"\n")
file.write("Test score: "+str(np.round(clf.score(X_test, Y_test), 5)))
file.close()
