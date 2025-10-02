import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt


df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg17Features")
print(df.head())

colnames = df.columns.tolist()
print(colnames)

Ten_Features = ['PLocID', 'Snaptime', 'Distance', 'Hang', 'Practice', 'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP']


TenFeaturesdf = df[Ten_Features]
print(TenFeaturesdf.shape)


colnames2= TenFeaturesdf.columns.to_list()
print(f"10 Featues and Label: {colnames2}")

X = TenFeaturesdf[['PLocID', 'Snaptime', 'Hang', 'Practice', 'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP']]
y = TenFeaturesdf['Distance']

regr = linear_model.LinearRegression()
regr.fit(X,y)

Coefficients = regr.coef_
Intercept = regr.intercept_

print(f"Formula: {Coefficients} + {Intercept}")

Score = regr.score(X,y)
print(f"Score for 10 features: {Score}")




TenFeaturesdf.to_csv('CleanedDistanceDataReg10Features', index = False)






