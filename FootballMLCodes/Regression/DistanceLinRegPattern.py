import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg17Features")
print(df.head())
colnames = df.columns.to_list()
print(colnames)



X = df.drop(columns=['Distance', 'PDate'], axis= 1)
print(X.columns.to_list())

y = df.Distance
print(y.head())

scores = {}


for column in X.columns:
    X_col = X[[column]]
    X_train, X_test, y_train, y_test = train_test_split(X_col, y, test_size=0.2, random_state=42)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test,y_test)

    scores[column] = score

    score_df = pd.DataFrame(scores.items(), columns=['Feature', 'R^2'])
    score_df = score_df.sort_values(by='R^2', ascending=False)

    print(score_df)



score_df.to_csv('LinRegCorrelation.csv', index=False)
    