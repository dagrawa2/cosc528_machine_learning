from __future__ import division
import pandas as pd

data = pd.read_csv("Data\\under5mortalityper1000.csv")

countries = list(data.iloc[:,0])
data = data.iloc[:,1:]
years = data.columns.values.tolist()

miss = data.isnull().sum()
empty_years = list(miss[miss==len(countries)].index.values)

data = data.iloc[:,:-10]
years = years[:-10]

miss = data.isnull().sum(axis=1)
empty_countries_indeces = list(miss[miss==len(years)].index.values)
#empty_countries = [countries[int(i)] for i in empty_countries_indeces]
#empty_countries = empty_countries[type(empty_countries)==str]

data = data.ix[[int(i) for i in list(data.index.values) if i not in empty_countries_indeces]]
print data.shape
