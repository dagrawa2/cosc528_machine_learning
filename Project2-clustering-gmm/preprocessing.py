from __future__ import division
import numpy as np
import pandas as pd
import pickle

# import data
data = pd.read_csv("Data\\under5mortalityper1000.csv")
# drop all empty rows;
# this means country name is also empty
data = data.dropna(axis=0, how="all")
# drop all empty columns
data = data.dropna(axis=1, how="all")

# list countries (first column)
countries_all = list(data.iloc[:,0].values)

# strip first column (countries)
data = data.iloc[:,1:]
# drop all empty rows;
# this drops country names with no entries
data = data.dropna(axis=0, how="all")

# data.shape: (211, 216)

# list all countries with nonempty data
countries = [countries_all[i] for i in list(data.index.values)]
# list all years with nonempty data
years = list(data.columns.values)

# count missing entries for each row
miss = list(data.isnull().sum(axis=1))
# the max is max(miss) = 190
# create a list of frequencies to visualize the distribution;
# freqs[i] is the number of countries with
# 10*i to 10*(i+1) missing values (right endpoint open)
freqs = [0]*20
for m in miss:
	freqs[m//10] += 1
# 27 out of 211 countries are missing at least
# 150 values out of 216 years
# all other countries are missing only 0 to 9 values

# drop the 27 countries with at least 150 missing values
indeces = []
for i in range(len(miss)):
	if miss[i] < 150:
		indeces.append(i)
data = data.iloc[indeces,:]
countries = [countries[i] for i in indeces]

# count missing values for each column
miss = list(data.isnull().sum(axis=0))
# there are no missing values

# save countries and years as lists
with open("Lists\\countries.list", "wb") as fp:
	pickle.dump(countries, fp)
with open("Lists\\years.list", "wb") as fp:
	pickle.dump(years, fp)

# standardize data and save it as numpy array
X = data.as_matrix()
X = (X-np.mean(X, axis=0))/np.std(X, axis=0)
np.save("Arrays\\X.npy", X)
