import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.columns.to_list())

X = df.drop(columns=['Efficiency', 'PDate', 'Height', 'Weight'])
print(X.columns.to_list())
y = df.Efficiency
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"RF Score: {score}")