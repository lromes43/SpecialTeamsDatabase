##Decision Tree Classifier


##Importing Libraries

import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


##Loading Dataset in
data = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')

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
print(data.shape) 
print("Reordered columns:", data.columns.to_list())

##Defining Variables

exclude_cols = [0]
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]
print(X.columns.to_list())

y = data.iloc[:, 0]  
y = y.values
print(y)


##Spitting Dataset
#use train_test_split method from sklearn.model_selection to split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)


##Defining Model
#using decision tree classifier from sklearn.tree create object for decision tree classifier

clf = DecisionTreeClassifier(random_state=1)

##Training the model
#apply the fit method to match the classifier to the training set of data

clf.fit(X_train, y_train)

##Making Predictions
#apply the predict method to the test data and use the trained model to create predictions

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

##Hyper paramter tuning with decision tree classifier using GridSearchCV
#hyperparamters are configuration settings that control the behavior of a decision tree model and affect its performance. Tuning can improve accuacy, reduce overfitting.

start = time.process_time() #time before

param_grid = {
    'max_depth': range(1,10,1),
    'min_samples_leaf' : range(1,20,2),
    'min_samples_split' : range(2,20,2),
    'criterion' : ["entropy", "gini"]
}

tree = DecisionTreeClassifier(random_state= 1)
grid_search = GridSearchCV(estimator= tree, param_grid= param_grid,
                           cv = 5, verbose= True)
grid_search.fit(X_train, y_train)
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)

end = time.process_time() # time after
CPU = end - start

## grid search cv evaluates the different parameters and list of possible values. 

##Visualzing Decision Tree
#plotting fwature importance obtaine to see greatest predictive power

tree_clf = grid_search.best_estimator_

plt.figure(figsize=(18,15))
plot_tree(tree_clf, filled= True, feature_names= X.columns.to_list(), 
          class_names=[str(cls) for cls in tree_clf.classes_]
          )

plt.title("Decision Tree Viz")
plt.show()

print(f"CPU Time: {CPU: .4f}")
