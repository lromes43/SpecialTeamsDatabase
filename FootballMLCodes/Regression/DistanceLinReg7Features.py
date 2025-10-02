import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg10Features")
print(df.head())


Seven_Features = ['PLocID', 'Snaptime', 'Hang', 'Practice', 'precipitation', 'Wind', 'Temp', 'Distance']
Seven_Featuresdf = df[Seven_Features]
print(Seven_Featuresdf.head())

colnames = Seven_Featuresdf.columns.to_list()
print(colnames)

X = Seven_Featuresdf[['PLocID', 'Snaptime', 'Hang', 'Practice', 'precipitation', 'Wind', 'Temp']]
y = Seven_Featuresdf['Distance']

regr = linear_model.LinearRegression()
regr.fit(X,y)

Coefficients = regr.coef_
Intercept = regr.intercept_

print(f"Coefficients: {Coefficients}")
print(f"Intercept: {Intercept}")

Score = regr.score(X,y)


print(f"Score: {Score}")

Seven_Featuresdf.to_csv('CleanedDistanceDataReg7Features', index= False)
