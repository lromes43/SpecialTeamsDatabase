import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, svm, tree, ensemble, naive_bayes, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


df = pd.read_csv("Football Data/Punt Data/PuntDataPulled.csv")
print(df.head())

#want to classify the punts as efficient or non-efficient
#use multitude of ML algorithms to see best one
#LogReg, SVM, RF, Decision Tree, GaussianNB, MultinomialNB


#define intial variables 

print(df.columns.to_list())

X = df.drop(['PDate', 'Height', 'Weight', 'Efficiency'], axis = 1)
print(X.head())

y = df[['Efficiency']]
print(y.head())

model_params = {
'LogReg':
    'model': 'LogisticRegression()',
    



}