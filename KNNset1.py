import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class MyKNN:
    def __init__(self, k=2):
        self.k = k
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            k_idx  = np.argsort(dists)[: self.k]
            k_lbls = self.y_train[k_idx]
            preds.append(Counter(k_lbls).most_common(1)[0][0])
        return np.array(preds)
    
        most_common = Counter(k_lbls).most_common()
        return most_common
    

# ---------- load and rearrange data ----------
csv = "/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv"
df  = pd.read_csv(csv)

cols = ['Efficiency','PLocID','Snaptime','Distance','Practice',
        'precipitation','Wind','Temp','H2F','PlayerIDLS','OP',
        'PDate','Hang','SnapLocID','Turf','Game','Grass','PlayerIDP']
df  = df[cols]

X = df.drop(columns='Efficiency')
y = df['Efficiency']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

start = time.process_time()
# ---------- sklearn KNN ----------
sk_clf = KNeighborsClassifier(n_neighbors=3)
sk_clf.fit(X_train, y_train)
sk_pred = sk_clf.predict(X_test)
end = time.process_time() # time after
CPU = end - start
print(" accuracy:", accuracy_score(y_test, sk_pred))






# ---------- simple 2‑D scatter (Snaptime vs Distance) ----------
cmap = ListedColormap(['#FF0000', '#00FF00'])   # 0 = red, 1 = green
X_np = X[['Snaptime','Distance']].values
y_np = y.values

plt.figure(figsize=(6,5))
plt.scatter(X_np[:,0], X_np[:,1],
            c=y_np, cmap=cmap, edgecolors='k', s=30)
plt.xlabel("Snaptime")
plt.ylabel("Distance")
plt.title("Snaptime vs Distance coloured by Efficiency")
plt.show()




logloss = log_loss(y_test, sk_pred)
roc_auc = roc_auc_score(y_test, sk_pred)
cm = confusion_matrix(y_test, sk_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [1,0]) # Use xgb_clf.classes_ for labels
disp.plot(cmap=plt.cm.Blues) # You can change the colormap
plt.title('Confusion Matrix for XGBoost Classifier (Test Set)')
plt.show() # Display the plot
print(f"CPU Time: {CPU:.4f} seconds")
print("Log Loss", logloss)
print(f"ROC AUC: {roc_auc:.4f}")