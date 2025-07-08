##Start with multiple linear regression

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels
from statsmodels import api as sm
from statsmodels import stats
from statsmodels.stats.stattools import durbin_watson


#Load in Data
csv = "/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv"
df  = pd.read_csv(csv)

print(df.columns)

df = df.drop(columns=['Efficiency'])

print(df.columns)

#Assign Variables
X = df[['PLocID', 'Snaptime', 'Practice', 'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP', 'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']]
y = df[['Distance']]

print(X.columns)
print(y.columns)

##Durbin Watson Statistic

X = sm.add_constant(X)

model = sm.OLS(y,X)
results = model.fit()

residuals = results.resid

dw_statistic = durbin_watson(residuals)

print(f"Durbin-Watson statistic: {dw_statistic}")

##Variance Inflation Factor

X = df[['PLocID', 'Snaptime', 'Practice', 'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP', 'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']]
y = df[['Distance']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42)

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


def DWS (test_residual ) : 
    n = len(test_residual)
    if n < 2:
        print("Warning! Not enough residuals need at least two")
        return None
    for i in range(1,n):
        numerator_sum =  (test_residual[i] - test_residual[i-1])**2
    for i in range(n):
        denomiator_sum = test_residual[i]**2
    if denomiator_sum == 0:
        print("Cannot Compute")
    return numerator_sum / denomiator_sum

dwsValue = DWS(test_residual)
if dwsValue is not None:
    print(f"Calculated DWS: {dwsValue}")









