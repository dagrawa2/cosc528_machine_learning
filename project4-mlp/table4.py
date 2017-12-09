from __future__ import division
import numpy as np
import pandas as pd

PC = [30,35,40,50,57]
data = pd.read_csv("Out\\out-pca.txt")
data = data.as_matrix()
pairs = zip(list(data[:,0]), list(data[:,1]))
D = {key:val for key,val in pairs}
vars = {pc:str(np.round(D[pc], 3)) for pc in PC}

table = open("Tables\\table4.txt", "w")

table.write("\\begin{tabular}{|c|c|c|c|c|} \\hline\n")
table.write("Num. of PCs & Frac. Var. Accounted & Loss & Training Acc. & Valid. Acc. \\\\ \\hline\n")

for pc in PC:
	file = open("Out\\out-pc"+str(pc)+".txt", "r")
	line = file.readlines()[-1]
	file.close()
	line = line.split(",")
	loss = line[1]
	acc_tr = line[2]
	acc_va = line[3]
	table.write(str(pc)+" & "+vars[pc]+" & "+loss+" & "+acc_tr+" & "+acc_va+" \\\\\n")

table.write("\\hline\n")
table.write("\\end{tabular}")

table.close()
