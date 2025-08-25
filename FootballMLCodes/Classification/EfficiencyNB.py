import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataPulled.csv")
print(df.head(3))

print(df.columns.to_list())


target = df['Efficiency']
input = df['SnapLocID']




#make df for each

NewTarget = pd.DataFrame(target)
print(NewTarget)
NewInput = pd.DataFrame(input)
print(NewInput)

#checking for number of na

print(NewTarget.isna().sum())
print(NewInput.isna().sum())


NewInput = NewInput.fillna(NewInput.median())
print(NewInput.isna().sum())


X_train, X_test, y_train, y_test = train_test_split(NewInput[['SnapLocID']], NewTarget[['Efficiency']], test_size=0.2)
model1 = GaussianNB()
model1.fit(X_train, y_train)
model1score = model1.score(X_test, y_test)

badsnap = model1.predict_proba(np.array([[3]]))
print(f"Bad Snap Prediction: {badsnap}")
print("Class labels:", model1.classes_)

goodsnap = model1.predict_proba(np.array([[1]]))
print(f"Good Snap Prediction: {goodsnap}")
print("Class labels:", model1.classes_)




