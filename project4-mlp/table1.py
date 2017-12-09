from __future__ import division
import numpy as np

table = open("Tables\\table1.txt", "w")

table.write("\\begin{tabular}{|c|c|c|c|} \\hline\n")
table.write("Hidden Neurons & Loss & Training Acc. & Valid. Acc. \\\\ \\hline\n")

H = [1,5,10,11,12,20,40]
for h in H:
	file = open("Out\\out-h"+str(h)+".txt", "r")
	line = file.readlines()[-1]
	file.close()
	line = line.split(",")
	loss = line[1]
	acc_tr = line[2]
	acc_va = line[3]
	table.write(str(h)+" & "+loss+" & "+acc_tr+" & "+acc_va+" \\\\\n")

table.write("\\hline\n")
table.write("\\end{tabular}")

table.close()
