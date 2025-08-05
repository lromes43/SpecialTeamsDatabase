import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.columns.to_list())

X = df.drop(columns=['Efficiency', 'Height', 'Weight', 'PDate'], axis=1)
print(X.columns.to_list())
y = df.Efficiency
print(y.head())

for columns in X.columns:
    X_col = X[[columns]]
    X_train, X_test, y_train, y_test = train_test_split(X_col, y, test_size=0.2, random_state=42)
    def best_C(X,y):
        scores = []
        for C_val in [0.001, 0.01, .1, 1, 10, 100]:
            model = SVC(C=C_val, max_iter=1000)
            mean_score = float(cross_val_score(model,X,y).mean())
            scores.append((C_val, mean_score))
        return scores
scores = best_C(X,y)
print(scores)


scoresdf = pd.DataFrame(scores, columns=['C-Value', 'Accuracy'])
plt.plot(scoresdf['C-Value'], scoresdf['Accuracy'])
plt.xlabel('C-Value')
plt.ylabel('Accuracy')
plt.title('C-Value vs Accuracy')
plt.show()