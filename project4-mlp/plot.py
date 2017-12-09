from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

suffix = "h20"

data = pd.read_csv("Out\\out-"+suffix+".txt")
data = data.as_matrix()
data = data[:121,:]

epochs = np.arange(data.shape[0])
loss = data[:,1]
acc_train = data[:,2]
acc_test = data[:,3]

fig = plt.figure()
st = fig.suptitle("Loss and Accuracy over Training Time")
ax1 = fig.add_subplot(2,2,1)
ax1.plot(epochs[:10], loss[:10], color="black")
ax1.set_ylabel("Loss")
ax2 = fig.add_subplot(2,2,2)
ax2.plot(epochs[10:], loss[10:], color="black")
ax2.axvline(x=100.5, color="gray", linestyle="-")
ax3 = fig.add_subplot(2,2,3)
ax3.plot(epochs[:10], acc_train[:10], color="blue")
ax3.plot(epochs[:10], acc_test[:10], color="red")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy")
ax4 = fig.add_subplot(2,2,4)
ax4.plot(epochs[10:], acc_train[10:], color="blue")
ax4.plot(epochs[10:], acc_test[10:], color="red")
ax4.axvline(x=100.5, color="gray", linestyle="-")
ax4.set_xlabel("Epoch")
#fig.subplots_adjust(hspace=0.75)
fig.savefig("Plots\\plot-"+suffix+".png", bbox_inches='tight')
#plt.show()
