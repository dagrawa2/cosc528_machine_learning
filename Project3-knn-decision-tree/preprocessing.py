from __future__ import division
import numpy as np
import pandas as pd

data = pd.read_csv("breast-cancer-wisconsin.data")
data = data.iloc[:,1:]
data = data.as_matrix()
for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		try:
			data[i,j] = float(data[i,j])
		except:
			data[i,j] = -1
data = data.astype(float)

col = data[:,5]
imp = np.mean(col[col>=0])
data[data<0] = imp
print imp

X = data[:,:-1]
y = data[:,-1]
y[y==2] = 0
y[y==4] = 1

np.save("Arrays\\X.npy", X)
np.save("Arrays\\y.npy", y)
