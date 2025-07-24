import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg7Features")
print(df.columns.to_list())

Five_Features = ['PLocID', 'Snaptime', 'Hang', 'Practice', 'precipitation', 'Distance']
Five_Featuresdf = df[Five_Features]
print(Five_Featuresdf.columns.to_list())
print(Five_Featuresdf.shape)


X = Five_Featuresdf[['PLocID', 'Snaptime', 'Hang', 'Practice', 'precipitation']]
y = Five_Featuresdf['Distance']

regr = linear_model.LinearRegression()
regr.fit(X,y)

Coefficients = regr.coef_
Intercept = regr.intercept_
Score = regr.score(X,y)


print(f"Coefficients: {Coefficients}")
print(f"Intercept: {Intercept}")
print(f"Score: {Score}")

Five_Featuresdf.to_csv('CleanedDistanceDataReg5Features', index = False)