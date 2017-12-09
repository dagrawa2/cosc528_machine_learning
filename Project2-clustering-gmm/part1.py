from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

# import preprocessed data
X = np.load("Arrays\\X.npy")

# perform SVD
U, s, V = np.linalg.svd(X)
# get array of eigenvalues
l = s**2

# get array of fractions of variance
# captured by the first k singular values
var_total = np.sum(l)
var_accum = 0
vars = []
for k in range(len(l)):
	var_accum += l[k]
	vars.append(var_accum)
vars = np.array(vars)/var_total

# print singular values and fraction of variance covered
# for the first few values of k
for k in range(10):
	print k, np.round(s[k], 5), np.round(vars[k], 5)

# plot s[k] against k
fig1 = plt.figure()
sub = fig1.add_subplot(1,1,1)
sub.plot(np.array(range(1,11)), s[:10])
sub.set_title("Singular Values")
sub.set_xlabel("$d$")
sub.set_ylabel("$s_d$")
fig1.savefig("Plots\\scree.png", bbox_inches='tight')
#plt.show()

# plot 100*vars[k] against k
fig2 = plt.figure()
sub = fig2.add_subplot(1,1,1)
sub.plot(np.array(range(1,11)), 100*vars[:10])
sub.set_title("Percentage of Variance Explained")
sub.set_xlabel("$d$")
sub.set_ylabel("Percentage")
fig2.savefig("Plots\\variance.png", bbox_inches='tight')
#plt.show()

# first three PCs account for about
# 91.8% of the variance:
print "Variance accounted by first three PCs: ", vars[2]

# construct matrix that projects onto first three PCs
W = V[:,:3]
# project the data onto first three PCs
Z = X.dot(W)

# save projection and projected data
np.save("Arrays\\W.npy", W)
np.save("Arrays\\Z.npy", Z)

# build scatter plot of data
# projected onto first two PCs
fig3 = plt.figure()
sub = fig3.add_subplot(1,1,1)
sub.scatter(Z[:,0], Z[:,1])
sub.set_title("Data Projected onto two Principal Components")
sub.set_xlabel("First PC")
sub.set_ylabel("Second PC")
fig3.savefig("Plots\\scatter.png", bbox_inches='tight')
#plt.show()
