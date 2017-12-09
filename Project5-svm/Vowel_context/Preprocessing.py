from __future__ import division
import numpy as np
import pandas as pd

data = pd.read_csv("Given\\vowel-context.data", delim_whitespace=True, header=None)
# data.shape = (990, 14)
data = data.as_matrix()

X = data[:,3:-1]
X = X.astype(float)
Y = data[:,-1]

np.save("Arrays\\X.npy", X)
np.save("Arrays\\Y.npy", Y)
