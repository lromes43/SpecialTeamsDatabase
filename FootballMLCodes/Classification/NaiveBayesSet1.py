##NaiveBayesSet1

##Actually Set 5
import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
from matplotlib import pyplot as plt
import time

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
# This step is good if you have a specific reason for reordering,
# e.g., for easier variable selection later or for aesthetic reasons.
# It doesn't typically affect model performance directly unless you're
# doing something position-dependent.
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
start = time.process_time()
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_prob = gnb.predict_proba(X_test)

end = time.process_time() # time after
CPU = end - start

logloss = log_loss(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob[:, 1])


print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))



cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [1,0]) # Use xgb_clf.classes_ for labels
disp.plot(cmap=plt.cm.Blues) # You can change the colormap
plt.title('Confusion Matrix for XGBoost Classifier (Test Set)')
plt.show() # Display the plot
print(f"CPU Time: {CPU:.4f} seconds")
print("Log Loss", logloss)
print(f"ROC AUC: {roc_auc:.4f}")