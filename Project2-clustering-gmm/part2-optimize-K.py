from __future__ import division
from algos import k_means, gaussian_mixture
import matplotlib.pyplot as plt
import numpy as np

Z = np.load("Arrays\\Z.npy")

variance = np.sum(Z**2)/(Z.shape[0]-1)

compressed_vars = []
for K in range(2, 15):
	model = k_means(K)
	model.train(Z)
	compressed_vars.append(model.compressed_variance/variance)
#	print "K = ", K
#	print "\n"
#	print "num_iters: ", model.num_iters
#	print "reconstruction error: ", model.reconstruction_error
#	print "cluster sizes: ", model.cluster_sizes
#	print "compressed variance (/variance): ", model.compressed_variance/variance
#	print "dunn index: ", model.dunn_index
#	print "\n"
compressed_vars = np.array(compressed_vars)

fig1 = plt.figure()
sub = fig1.add_subplot(1,1,1)
sub.plot(np.array(range(2,15)), 100*compressed_vars)
sub.set_title("Compressed Variance")
sub.set_xlabel("$K$")
sub.set_ylabel("Percentage")
fig1.savefig("Plots\\compressed-variance.png", bbox_inches='tight')
#plt.show()
