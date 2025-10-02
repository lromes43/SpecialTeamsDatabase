import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.columns.to_list())

X = df.drop(columns=['Efficiency', 'PDate', 'Height', 'Weight'])
print(X.columns.to_list())
y = df.Efficiency
print(y.head())



model = RandomForestClassifier(n_estimators=10, criterion='gini',min_samples_split=5, min_samples_leaf=3, random_state=42)
new_score = float(cross_val_score(model,X,y).mean())
print(f"Updated Score: {new_score}")


with open('RandomForestClassifierTrained', 'wb') as f:
    pickle.dump(model,f)



