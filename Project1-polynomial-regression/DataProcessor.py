from __future__ import division
import numpy as np
import re

# import lines of data
print "Reading data . . . "
file = open("auto-mpg.data", "r")
lines = file.readlines()
file.close()

print "Creating data array"
data = []
for line in lines:
	# replace missing values "?" with "-1" for the horsepower feature;
	# this will allow us to convert the data into a numpy array
	# note we know horsepower can never be negative
	line = line.replace("?", "-1")
	# discard the car name feature (the last column)
	# it is unique for each observation
	L = re.split("\s+", line.split("\t")[0])
	# convert strings to floating point
	data.append([float(l) for l in L])

# convert list of lists to numpy array
data = np.asarray(data)

# impute the six missing values for the horsepower feature
# with the average of all other values
# note missing values are represented as -1
print "Imputing missing values . . . "
impute_value = (np.sum(data[:,3])+6)/(data.shape[0]-6)
count = 0
for i in range(data.shape[0]):
	if data[i,3] == -1:
		data[i,3] = impute_value
		count += 1
print str(count) + "missing values imputed"

# Save data array
print "Saving data array"
np.save("Data.npy", data)
print "Data written to Data.npy"

# standardize data
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data-mean)/std

#np.save("DataMean.npy", mean)
#np.save("DataStd.npy", std)
print "Saving standardized data . . . "
np.save("DataStandardized.npy", data)
print "Standardized data written to DataStandardized.npy"