# This code is a modification of the code found at
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from tools import MidpointNormalize

M = [(0.4, 0.85), (0.4, 0.85)]

for num in range(1, len(M)+1):
	C_range = np.load("Plot_data\\Plot_"+str(num)+"\\C_range.npy")
	gamma_range = np.load("Plot_data\\Plot_"+str(num)+"\\gamma_range.npy")
	scores = np.load("Plot_data\\Plot_"+str(num)+"\\scores.npy")

#	vmin, midpoint = M[num-1]
	vmin, midpoint = np.min(scores), np.mean(scores)

	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=vmin, midpoint=midpoint))
	plt.xlabel("$\gamma$")
	plt.ylabel("$C$")
	plt.colorbar()
	plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
	plt.yticks(np.arange(len(C_range)), C_range)
	plt.title("Refinement = "+str(num-1))
	plt.savefig("Plots\\plot_"+str(num)+".png", bbox_inches='tight')
	#plt.show()
