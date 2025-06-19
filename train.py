import numpy as np
import pandas as pd
#import KNN
from collections import Counter
import sklearn
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap


cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


data = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')

#Check Number of Missing Values
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


exclude_cols = [0] # This refers to 'Efficiency' after reordering
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]

#print(X.columns.to_list()) # Uncomment to verify your X columns

y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

X_np = X.values
y_np = y.values

plt.figure(figsize=(6,5))
plt.scatter(X_np[:,1],
            X_np[:,2],
            c = y_np,
            cmap= cmap,
            edgecolors= 'k',
            s = 20)
plt.xlabel('Snaptime')
plt.ylabel('Distance')
plt.title('Snaptime vs Distance')
plt.show()

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)

