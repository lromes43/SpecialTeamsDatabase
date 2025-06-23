import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


csv = "/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv"
df  = pd.read_csv(csv)

cols = ['Efficiency','PLocID','Snaptime','Distance','Practice',
        'precipitation','Wind','Temp','H2F','PlayerIDLS','OP',
        'PDate','Hang','SnapLocID','Turf','Game','Grass','PlayerIDP']
df  = df[cols]

X = df.drop(columns=['Efficiency', 'H2F','PlayerIDLS','OP','PDate','Hang','SnapLocID','Turf','Game','Grass','PlayerIDP'])
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)



regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.coef_)

score = regr.score

test = regr.fit(X_test, y_test)
print(test.coef_)
print(test.score(X_test, y_test))
