##Logistic Regression
##Loading in Libraries and Modules
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model 
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import time



##Loading Dataset in
data = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')
##Doing print of the head to make sure its in and correct
#print(data.head())

##Check Number of Missing Values
missing = data.isnull().sum()
print(missing)

##Get Dimensions of DF
dimensions = data.shape
print(dimensions)

##Defining Variables
exclude_cols = [11, 12, 13]
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]
print(X)
print(X.shape)

y = data.iloc[:, 10]  
y = y.values
print(y)
print(y.shape)


#Using LogisticRegression() method to create logistic regression object
#Object has method called fit() that takes the independent and dependent values as parameters and fills the regression object with data that describes the relarionship

logr = linear_model.LogisticRegression()
logr.fit(X, y)

##Predict 

new_sample = np.array([96, 1, 58, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0])
new_sample = new_sample.reshape(1, -11)

predicted = logr.predict(new_sample)
print(predicted)


##Coefficient
#Coefficent is the expected change in the log odds of having the outcome per unto change in X
##Tells when feature increases by 1 unit what are odds of positive increase or decrase by that factor

log_odds = logr.coef_
odds = np.exp(log_odds)
print(odds)


##Probability

##Coefficient and intercept values can be used to find prob of being efficent
#create function that uses the models coefficent and intercept values to return a new values
#new value represents probability of efficiency

def logit2prob (logr, x):
    log_odds = X.values @ logr.coef_.T + logr.intercept_
    odds = np.exp(log_odds)
    probaility = odds / (1 + odds)
    return(probaility)

print(logit2prob(logr, X))


##Metrics


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = 42)


logr.fit(X_train, y_train)

y_prob = logr.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
logloss = log_loss(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")


from collections import Counter
print(Counter(y))
print(Counter(y_test))


