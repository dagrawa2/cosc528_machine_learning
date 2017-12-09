from __future__ import division
import numpy as np
import pandas as pd

data = pd.read_csv("Given\\ionosphere.data", header=None)
# data.shape = (351, 35)
data = data.as_matrix()

X = data[:,:-1]
X = X.astype(float)

Y = np.zeros((data.shape[0]))
Y[data[:,-1]=="g"] = 1

np.save("Arrays\\X.npy", X)
np.save("Arrays\\Y.npy", Y)
