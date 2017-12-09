from __future__ import division
import numpy as np

table = open("Tables\\table2.txt", "w")

table.write("\\begin{tabular}{|c|c|c|} \\hline\n")
table.write("Output Func. & Training Acc. & Valid. Acc. \\\\ \\hline\n")

filenames = ["Out\\out-h11linear.txt", "Out\\out-h11.txt", "Out\\out-h11softmax.txt"]
outputs = ["linear", "logistic", "softmax"]

for filename, output in zip(filenames, outputs):
	file = open(filename, "r")
	line = file.readlines()[-1]
	file.close()
	line = line.split(",")
	acc_tr = line[2]
	acc_va = line[3]
	table.write(output+" & "+acc_tr+" & "+acc_va+" \\\\\n")

table.write("\\hline\n")
table.write("\\end{tabular}")

table.close()