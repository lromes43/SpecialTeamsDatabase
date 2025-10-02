import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
X = df.drop(columns=['Efficiency', 'PDate', 'Height', 'Weight'])
print(X.columns.to_list())
y = df.Efficiency
print(y.head())

LogRegCV = cross_val_score(LogisticRegression(),X, y).mean()
SVMCV = cross_val_score(SVC(), X, y).mean()
RFCV = cross_val_score(RandomForestClassifier(), X, y).mean()


print(f"Log Reg CV Score: {LogRegCV}")
print(f"SVM CV Score: {SVMCV}")
print(f"RF CV Score: {RFCV}")

#RF is best model overall