##KNN Set 1


#Importing Libraries
import numpy as np
import pandas as pd
from collections import Counter
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '##00FF00', '#0000FF'])

#I#mporting Data
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

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(x1-x2)**2)
    return distance

class KNN:
    def __init__(self, k=3): ##setting to 3 to start
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predictions = [self._predict(X) for X in X]
        return predictions


    def _predict(Self, x):
        #Compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]




        #get the closest K
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]

        
        #determine label with majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


plt.figure()
plt.scatter(X[:,2], X[:,3], c = y, cmap=cmap, edgecolors= 'k', s = 20)
plt.show()

