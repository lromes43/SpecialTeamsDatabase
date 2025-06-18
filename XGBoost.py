import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import pipeline # Not used in the provided snippet
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
from matplotlib import pyplot as plt
import time

##explore and prep data

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

##Defining Variables
# This is a bit convoluted. If 'Efficiency' is always the first column after reordering,
# you can select X and y more directly.
# Also, it's good practice to keep X as a DataFrame and y as a Series for clarity.
exclude_cols = [0] # This refers to 'Efficiency' after reordering
include_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
X = data.iloc[:, include_cols]
#print(X.columns.to_list()) # Uncomment to verify your X columns

y = data.iloc[:, 0]
# y = y.values # Converting to numpy array is fine, but keeping as Series can be helpful sometimes
              # XGBoost handles both pandas Series/DataFrame and numpy arrays.


#Separating Data
start = time.process_time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
prediction = model.predict(X_test)

y_pred  = model.predict(X_test)          # hard labels  0/1
y_prob  = model.predict_proba(X_test)

end = time.process_time() # time after
CPU = end - start
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, prediction)


print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, prediction, target_names= ['Inefficient', 'Efficient']))


cm = confusion_matrix(y_test, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


LogLoss = log_loss(y_test, prediction)
roc_auc  = roc_auc_score(y_test, y_prob[:,1]) 
print("Log Loss:", LogLoss)
print(f"ROC AUC: {roc_auc:.4f}")


cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("=== 5-Fold Cross-Validation Accuracy Scores ===")
print("Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
print("Std Dev:", np.std(cv_scores))
print(f"CPU Time: {CPU:.4f} seconds")

