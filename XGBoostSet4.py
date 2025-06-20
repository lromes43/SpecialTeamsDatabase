##XGBoostSet2

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
from matplotlib import pyplot as plt
import time


csv = "/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv"
df  = pd.read_csv(csv)

cols = ['Efficiency','PLocID','Snaptime','Distance','Practice',
        'precipitation','Wind','Temp','H2F','PlayerIDLS','OP',
        'PDate','Hang','SnapLocID','Turf','Game','Grass','PlayerIDP']
df  = df[cols]

X = df.drop(columns=['Efficiency', 'PlayerIDP', 'Grass', 'Game', 'Turf', 'SnapLocID', 'Hang', 'PDate', 'OP', 'PlayerIDLS', 'H2F', 'Temp', 'Wind', 'precipitation'])
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

#Separating Data
start = time.process_time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

xgb_clf = XGBClassifier(
    n_estimators = 500,
    learning_rate = 0.05,
    use_label_encoder = False,
    eval_metric = "logloss",
    early_stopping_rounds = 5,
    n_jobs = -1,
    random_state = 42
)

xgb_clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

end = time.process_time() # time after
CPU = end - start
from sklearn.metrics import accuracy_score, classification_report


pred_test = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred_test)
test_score = accuracy_score(pred_test, y_test)
print("Test Score: ", np.round(test_score, 2))

print(f"CPU Time: {CPU:.4f} seconds")

pred_train = xgb_clf.predict(X_train)
train_score = accuracy_score(pred_train, y_train)
print("Train Score: ", np.round(train_score, 2))


logloss_test = log_loss(y_test, pred_test)
print(f"Test Log Loss: {np.round(logloss_test, 4)}")
roc_auc = roc_auc_score(y_test, pred_test)
print(f"ROC AUC: {roc_auc:.4f}")

cm = confusion_matrix(y_test, pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_clf.classes_) # Use xgb_clf.classes_ for labels
disp.plot(cmap=plt.cm.Blues) # You can change the colormap
plt.title('Confusion Matrix for XGBoost Classifier (Test Set)')
plt.show() # Display the plot



'''
fpr, tpr, thresholds = roc_curve(y, pred_test)
auc_score = roc_auc_score(y, pred_test)
print(f"Area Under the Curve (AUC): {auc_score:.3f}")

'''
