from __future__ import division
import numpy as np
import os

filenames = os.listdir("Out")

file = open("Table\\table.txt", "w")
file.write("\\begin{tabular}{|c|c|c|c|} \\hline\n")
file.write("Refinement & $c$ & $\\gamma$ & Accuracy \\\\ \\hline\n")

for filename in filenames[:-1]:
	i = str(int(filename[4])-1)
	out = open("Out\\"+filename, "r")
	lines = out.readlines()
	out.close()
	lines = [line[:-1] if line[-1] == "\n" else line for line in lines]
	c = lines[1][3:]
	gamma = lines[2][7:]
	score = lines[3][12:]
	file.write(i+" & "+c+" & "+gamma+" & "+score+" \\\\\n")

out = open("Out\\"+filenames[-1], "r")
lines = out.readlines()
out.close()
lines = [line[:-1] if line[-1] == "\n" else line for line in lines]
c = lines[1][3:]
gamma = lines[2][7:]
score = lines[3][12:]
file.write("test & "+c+" & "+gamma+" & "+score+" \\\\\n")
file.write("\\hline\n")
file.write("\\end{tabular}")
file.close()
