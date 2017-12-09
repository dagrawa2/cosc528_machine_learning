from __future__ import division
import numpy as np

data = np.load("Data.npy")
mpg = data[:,0]
print "Stats on the mpg column:\n"
print "min: ", np.min(mpg)
print "max: ", np.max(mpg)
print "mean: ", np.mean(mpg)
print "std: ", np.std(mpg)
