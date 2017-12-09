from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from algos import *
from tools import *

X = np.load("Arrays\\X.npy")
y = np.load("Arrays\\y.npy")

X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.8)

np.save("Arrays\\X_train.npy", X_train)
np.save("Arrays\y_train.npy", y_train)
np.save("Arrays\X_test.npy", X_test)
np.save("Arrays\y_test.npy", y_test)
