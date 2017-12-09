from __future__ import division
from algos import k_means
import numpy as np

def random_cluster(center, radius, num):
	r = np.random.uniform(0, radius, num)
	theta = np.random.uniform(0, 2*np.pi, num)
	return np.asarray(center) + np.array([r*np.cos(theta), r*np.sin(theta)]).T

N = 1000
C_1 = random_cluster([-3, 0], 1, N//3)
C_2 = random_cluster([3, 0], 1, N//3)
C_3 = random_cluster([0, 3], 1, N//3)
means = np.array([np.mean(C_1, axis=0), np.mean(C_2, axis=0), np.mean(C_3, axis=0)])

X = np.concatenate((C_1, C_2, C_3), axis=0)

model = k_means(3)
model.train(X)

print "num_iters: ", model.num_iters
print "reconstruction error: ", model.reconstruction_error
print "min intracluster distance: ", model.min_intracluster_distance
print "max intercluster distance: ", model.max_intercluster_distance
print "dunn index: ", model.dunn_index
for k in range(model.K):
	print "cluster "+str(k+1)+":"
	print "true mean: ", np.round(means[k,:], 5)
	print "model mean: ", np.round(model.means[k,:], 5)
