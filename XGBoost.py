###XGBoost

#Gradient Boosting: ML algorithm that sequentially ensembles weak predictive models into a single stronger predictive model, most common model is decision tree
#XGBoost: overcomes decision tree downfalls, more regularized, objective function of XGboost has a regularixation term added to the loss function, more scalable as well with memoery and cache optimization as well as distributed computing.

import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection 
from sklearn.model_selection import train_test_split

##explore and prep data

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
exclude_cols = [0]
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]
#print(X.columns.to_list())

y = data.iloc[:, 0]  
y = y.values

#Separating Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state= 42)


##Build Pipeline of Data
from sklearn import pipeline
from sklearn.pipeline import Pipeline
import category_encoders
from category_encoders import target_encoder
from category_encoders.target_encoder import TargetEncoder
import xgboost
from xgboost import XGBClassifier

estimators = [
    ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state = 42))
]

pipe = Pipeline(steps= estimators)
print(pipe)


##Set up hyperparameter tuning
import skopt
