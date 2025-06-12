## SVM
#goal is to find best boundary known as a hyperplane that seperates the diff classes in data
#hyperplane: decisio boundary separating different classes in feature space and is represented by the euqation; wx + b =0
#Support vectors: closest data points to the hyperplane, crucial for determining the hyperplane and margin
#Margin: distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification performance
#Kernel: function that maps data to a higher dimensional space enabling SVM to handle non-linearly seperate data
#Hard Margin: max -margin hyerplane that perfectly separates the data without misclassifications
#Soft margin: allows some misclassifications by introducting slack variables, balancing margin maximization and misclassification penalties when data is not perfectly seperable
#C: regularization term balancing margin max and misclassifciation penalties. Higher C value forces stricter penalties for misclassifications
#Hinge Loss: loss function penalizing misclassified points or margin violations and is combined with regularization in SVM
#Dual Problem: involves solving for lagrange multipliers associated with support vectors, facilitating the kernel trick and efficient computation

import sklearn
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn import inspection
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

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
#print(X.columns.to_list())

y = data.iloc[:, 0]  
y = y.values
#print(y)

##Building the model 
svm = SVC(kernel= "rbf", gamma= 0.5, c = 1)

#train the model
svm.fit(X,y)

##Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    cmap = plt.cm.Spectral,
    alpha = 0.8,
    xlabel= data.iloc[:, 17],
    ylabel= "efficiency"

)

##scatterplot
plt.scatter(X[:, 0], X[:, 1], 
            c=y, 
            s=20, edgecolors="k")
plt.show()