from __future__ import division
from algos import gaussian_mixture
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

K = 6

color_choices = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

X = np.load("Arrays\\X.npy")
Z = np.load("Arrays\\Z.npy")
with open("Lists\\countries.list", "rb") as f:
	countries = pickle.load(f)

variance = np.sum((X-np.mean(X, axis=0, keepdims=True))**2)/(X.shape[0]-1)

model = gaussian_mixture(K)
model.train(X, eps=10**-9, suff_cluster_variance=0.98)

clusters = model.cluster_hard(X)
cluster_sizes = np.array([len(clusters[clusters==k]) for k in range(model.K)])
file = open("Clusters\\GMM\\X\\output.txt", "w")
file.write("K = "+str(K)+"\r\n\r\n")
file.write("iterations: "+str(model.num_iters)+"\r\n")
file.write("fraction of variance explained: "+str(model.compressed_variance/variance)+"\r\n")
file.write("max intracluster distance: "+str(model.max_intracluster_distance)+"\r\n")
file.write("min intercluster distance: "+str(model.min_intercluster_distance)+"\r\n")
file.write("dunn index: "+str(model.dunn_index)+"\r\n")
file.write("hard cluster sizes: [")
for k in range(K):
	file.write(str(cluster_sizes[k])+", ")
file.seek(-2, os.SEEK_CUR)
file.write("]\r\n")
file.close()

for k in range(K):
	L = [countries[i] for i in range(len(countries)) if clusters[i]==k]
	L.sort()
	file = open("Clusters\\GMM\\X\\cluster_"+str(k+1)+".txt", "w")
	file.write("cluster size: "+str(cluster_sizes[k])+"\r\n")
	file.write("color in plot: "+color_choices[k]+"\r\n\r\n")
	for country in L:
		file.write(country+"\r\n")
	file.close()

colors = [color_choices[k] for k in list(clusters)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Z[:,0], Z[:,1], c=colors)
ax.set_title("Clusters")
#ax.set_xlabel("First PC")
#ax.set_ylabel("Second PC")
fig.savefig("Clusters\\GMM\\X\\plot.png", bbox_inches='tight')
#plt.show()
