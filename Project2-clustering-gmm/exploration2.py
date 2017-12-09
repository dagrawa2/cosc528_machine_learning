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

miss.sort()
#print miss[-28:]

print 100*151/216