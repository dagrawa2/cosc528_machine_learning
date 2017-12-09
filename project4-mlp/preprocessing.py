from __future__ import division
import numpy as np
import pandas as pd

data = pd.read_csv("Given\\spambase.data")
data = data.as_matrix()
X = data[:,:-1]
Y = data[:,-1]
Y = Y.reshape((len(Y), 1))

np.save("Arrays\\X.npy", X)
np.save("Arrays\\Y.npy", Y)
