import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg5Features")
print(df.columns.to_list())

Three_Features = ['PLocID', 'Snaptime', 'Hang', 'Distance']
Three_Featuresdf = df[Three_Features]

print(Three_Featuresdf.columns.to_list)
print(Three_Featuresdf.shape)

X = Three_Featuresdf[['PLocID', 'Snaptime', 'Hang']]
y = Three_Featuresdf['Distance']

regr = linear_model.LinearRegression()
regr.fit(X,y)

Coefficient = regr.coef_
Intercept = regr.intercept_
Score = regr.score(X,y)

print(f"Coefficients: {Coefficient}")
print(f"Intercept: {Intercept}")
print(f"Score: {Score}")
