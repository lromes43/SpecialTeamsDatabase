##Start with multiple linear regression

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels
from statsmodels import api as sm
from statsmodels import stats
from statsmodels.stats.stattools import durbin_watson


#Load in Data
csv = "/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataPulled.csv"
df  = pd.read_csv(csv)

print(df.columns)
print(df.head)

df = df.drop(columns=['Efficiency'])

print(df.columns)

#Assign Variables
X = df[['PLocID', 'Snaptime', 'Practice', 'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP', 'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']]
y = df[['Distance']]

print(X.columns)
print(X.shape)
print(y.columns)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X,y)

Coefficients = regr.coef_
print(f"Coefficients: {Coefficients}")

#predictedDist = regr.predict([[3,.7,1, 1.00, 5, 67, .7,41, 2.0, 2024-08-02, 4, 1, 0, 0, 1, 96])
#print(predictedDist)





'''

##Durbin Watson Statistic

X = sm.add_constant(X)

model = sm.OLS(y,X)
results = model.fit()

residuals = results.resid

dw_statistic = durbin_watson(residuals)

print(f"Durbin-Watson statistic: {dw_statistic}")
#At 5% significance level dl = 1.599, du = 1.943, DW was between
#n = 200
#k = 16

#DW proved to be inconclusive, doing Breusch-Godrey Test






##Variance Inflation Factor

X = df[['PLocID', 'Snaptime', 'Practice', 'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP', 'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']]
y = df[['Distance']]



print(X.shape)
print(y.shape)
test = LinearRegression()

resultss = test.fit(X_train, y_train)

y_pred = resultss.predict(X_test)
print(y_pred)


print(y_pred.shape)
print(y_train.shape)

y_pred_train = test.predict(X_train)
y_pred_test = test.predict(X_test)

test_residual = y_test - y_pred_test
print(test_residual)

print(isinstance(test_residual, pd.DataFrame))




'''















