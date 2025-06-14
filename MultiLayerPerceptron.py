##Multilayer Perceptrons (Neural Networks)
##See google drive for diagram

import pandas as pd
import numpy as np

data = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')

##Check Number of Missing Values
missing = data.isnull().sum()
print(missing)

##Get Dimensions of DF
dimensions = data.shape
print(dimensions)

##Getting Names of Columns

column_names = data.columns.to_list()
print("Original columns:", data.columns.to_list())

##Changing column order
new_order = ['Efficiency', 'PLocID', 'Snaptime', 'Distance', 'Practice',
             'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP',
             'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']

# Apply reorder
data = data[new_order]

# Confirm final structure
print(data.shape) 
print("Reordered columns:", data.columns.to_list())

##Defining Variables
exclude_cols = [0,]
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]
print("X-values : ", X.columns.to_list())

y = data.iloc[:, 0]  
y = y.values

print(X["PlayerIDP"].head())
