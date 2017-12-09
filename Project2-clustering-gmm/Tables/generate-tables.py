import numpy as np
import os

Clusters_KMC_path = os.getcwd()+"\\..\\Clusters\\KMC"
Clusters_GMM_path = os.getcwd()+"\\..\\Clusters\\GMM"

def get_numbers(filename):
	with open(filename, "r") as fp:
		lines = fp.readlines()
		lines = [line.strip("\r\n") for line in lines]
		lines = [line for line in lines if len(line) > 0]
		lines = lines[1:-1]
		lines = [line.split(": ")[-1] for line in lines]
		nums = [lines[0]]
		temp = np.round(np.array([float(line) for line in lines[1:]]), 4)
		for i in range(len(temp)):
			nums.append(str(temp[i]))
	return nums

labels = ["iterations", "compressed var.", "max intracluster dist.", "min intercluster dist.", "Dunn index"]

nums_X = get_numbers(Clusters_KMC_path+"\\X\\output.txt")
nums_Z = get_numbers(Clusters_KMC_path+"\\Z\\output.txt")
nums_Z2D = get_numbers(Clusters_KMC_path+"\\Z2D\\output.txt")

file = open("table-KMC.txt", "w")
file.write("\\begin{tabular}{|c|c|c|c|} \\hline\r\n")
file.write("\quad & $X$ & $Z$ & $Z_{2D}$ \\\\ \\hline\r\n")
for i in range(len(labels)):
	file.write(labels[i]+" & "+nums_X[i]+" & "+nums_Z[i]+" & "+nums_Z2D[i]+" \\\\ \\hline\r\n")
file.write("\\end{tabular}\r\n")
file.close()

nums_X = get_numbers(Clusters_GMM_path+"\\X\\output.txt")
nums_Z = get_numbers(Clusters_GMM_path+"\\Z\\output.txt")
nums_Z2D = get_numbers(Clusters_GMM_path+"\\Z2D\\output.txt")

file = open("table-GMM.txt", "w")
file.write("\\begin{tabular}{|c|c|c|c|} \\hline\r\n")
file.write("\quad & $X$ & $Z$ & $Z_{2D}$ \\\\ \\hline\r\n")
for i in range(len(labels)):
	file.write(labels[i]+" & "+nums_X[i]+" & "+nums_Z[i]+" & "+nums_Z2D[i]+" \\\\ \\hline\r\n")
file.write("\\end{tabular}\r\n")
file.close()
