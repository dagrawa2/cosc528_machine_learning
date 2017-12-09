from __future__ import division
import numpy as np
import pandas as pd

data_train = pd.read_csv("Given\\sat.trn", delim_whitespace=True, header=None)
data_test = pd.read_csv("Given\\sat.tst", delim_whitespace=True, header=None)

data = np.concatenate((data_train.as_matrix(), data_test.as_matrix()), axis=0)
# data.shape = (6435, 37)

X = data[:,:-1]
X = X.astype(float)
Y = data[:,-1]

np.save("Arrays\\X.npy", X)
np.save("Arrays\\Y.npy", Y)
