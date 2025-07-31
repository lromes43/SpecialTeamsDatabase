import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg17Features")
print(df.head())
print(df.columns.to_list())


X = df.drop(columns=['Distance', 'PDate'], axis= 1)
print(X.columns.to_list())

y = df.Distance
print(y.head())

scores = {}

for columns in X.columns:
    X_col = X[[columns]]
    X_train, X_test, y_train, y_test = train_test_split(X_col, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=0.01, max_iter=10000)
    model.fit(X_train, y_train)


    score = model.score(X_test, y_test)
    scores[columns] = score

    df_new = pd.DataFrame(scores.items(), columns=['Features', 'R^2'])
    df_new = df_new.sort_values(by='R^2', ascending=False)

    print(df_new)




