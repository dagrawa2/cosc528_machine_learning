from __future__ import division
from algos import gaussian_mixture
import numpy as np

n = 1000
C_1 = np.random.multivariate_normal(np.array([-3,0]), 0.5*np.eye(2), int(0.2*n))
C_2 = np.random.multivariate_normal(np.array([3,0]), 1.5*np.eye(2), int(0.3*n))
C_3 = np.random.multivariate_normal(np.array([0,3]), np.eye(2), int(0.5*n))
X = np.concatenate((C_1, C_2, C_3), axis=0)

variance = np.mean(np.sum((X-np.mean(X, axis=0, keepdims=True))**2, axis=1))

model = gaussian_mixture(3)
model.train(X, eps=10**-9)
print "num_iters: ", model.num_iters
print "compressed variance (/variance): ", model.compressed_variance/variance
print "mean intracluster variance (/variance): ", model.pi.dot(model.intracluster_variances)/variance
for k in range(3):
	print "Cluster "+str(k+1)+":"
	print "pi_"+str(k+1)+" = ", np.round(model.pi[k], 5)
	print "mu_"+str(k+1)+" = ", np.round(model.mu[k,:], 5)
	print "sigma_"+str(k+1)+" = ", np.round(model.sigma[k,:,:], 5)
