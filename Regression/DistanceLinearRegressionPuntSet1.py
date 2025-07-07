##Start with multiple linear regression

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt


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

##X_train, X_test, y_train, y_test = train_test_split(
   ##     X, y, test_size=0.12, random_state=42)

#Using the linearregression method to create a linear regression object

regr = linear_model.LinearRegression()
regr.fit(X,y)

slope, intercept, r, p, std_err = stats.linregress(X, y)

def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, x))

print(mymodel)





