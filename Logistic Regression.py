##Logistic Regression
##Loading in Libraries and Modules
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model 
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import time
import matplotlib.pyplot as plt



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

##Getting Names of Columns

column_names = data.columns.to_list()
print("Original columns:", data.columns.to_list())

##Changing column order
new_order = ['Efficiency', 'PLocID', 'Snaptime', 'Distance', 'Practice',
             'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP',
             'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']

# Apply reorder
data = data[new_order]

# Confirm final structure
print(data.shape)  # Should be (331, 18)
print("Reordered columns:", data.columns.to_list())

##Defining Variables
exclude_cols = [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]
print(X)
print(X.shape)

y = data.iloc[:, 0]  
y = y.values
print(y)
print(y.shape)

#Using LogisticRegression() method to create logistic regression object
#Object has method called fit() that takes the independent and dependent values as parameters and fills the regression object with data that describes the relarionship

logr = linear_model.LogisticRegression(max_iter=1000)
logr.fit(X, y)

##Predict 

new_sample = np.array([0, 0, 0])
new_sample = new_sample.reshape(1, 3)

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

start = time.process_time() #time before
def logit2prob (logr, x):
    log_odds = X.values @ logr.coef_.T + logr.intercept_
    odds = np.exp(log_odds)
    probaility = odds / (1 + odds)
    return(probaility)

print(logit2prob(logr, X))


##Metrics

##splits data into 90% training (x_train, y_train)
##10% test(x_test, y_test)
###random state ensures same split
#42 for odd, 43 for even
#change test size based on runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15 , random_state = 42) 

##fits the logistic regression model to training set
logr.fit(X_train, y_train)

##Predicted Probabilities
#predict_proba returns prob of each class
#[;, 0] = prob of class 0
#[;, 1] = prob of class 1
#doing 1 bc seeing how well can do it
y_prob = logr.predict_proba(X_test)[:, 1]



##Converting Probabilities to class predictions
#if predicted prob greater or equal .5 predict 1, else do 0
y_pred = (y_prob >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred) #% total correct predictions
precision = precision_score(y_test, y_pred) #of all predicted 1s, how many were correct
recall = recall_score(y_test, y_pred) #of all actual 1s, how many did we catch
f1 = f1_score(y_test, y_pred) #harmonic mean of precision and recall
roc_auc = roc_auc_score(y_test, y_prob) #how well model separates two classes overall
logloss = log_loss(y_test, y_prob) #penalizes incorrect more heaving, lower better
end = time.process_time() # time after
CPU = end - start
CVScores = cross_val_score(logr, X, y, cv = 5, scoring = 'accuracy')
##Confusion Matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])



#Displaying matrix
cm_display.plot()
plt.show()


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"CPU Time: {CPU: .4f}")

print(f"Cross Val Accuacy scores: {CVScores}")
print(f"Mean CV accuacy: {np.mean(CVScores)}")
print(f"St Dev CV: {np.std(CVScores)}")

from collections import Counter
print(Counter(y))
print(Counter(y_test))




print(X.columns.to_list()[:5])




