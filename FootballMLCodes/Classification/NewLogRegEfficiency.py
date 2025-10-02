import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.head())
print(df.columns.to_list())

X = df.drop(columns=['PDate', 'Efficiency', 'Height', 'Weight'], axis= 1)
print(X.columns.to_list())

y = df.Efficiency
print(y.head())

for col in X.columns:
    print(f"{col}: {X[col].isna().sum()}")







for columns in X.columns:
    X_col = X[[columns]]
    X_train, X_test, y_train, y_test = train_test_split(X_col, y, test_size=0.2, random_state=42)
    def best_c_value (X,y):
        scores = []
        for c_val in [0.001, 0.01, 1, 10, 100]:
            model = LogisticRegression(C= c_val, max_iter=10000)
            mean_score = float(cross_val_score(model, X, y).mean())
            scores.append((c_val, mean_score))
        return scores
    
    scores = best_c_value(X,y)

print(scores)



scores_df = pd.DataFrame(scores, columns=['C-Value', 'Accuracy'])
import matplotlib.pyplot as plt

plt.plot(scores_df['C-Value'], scores_df['Accuracy'], color = 'red')
plt.xlabel('C-Value')
plt.ylabel('Mean Accuracy')
plt.title('C-Value vs Mean Accuracy')
plt.show()

#Optimum C value is 10 before overfitting



    


'''


for columns in X.columns:
    X_col = X[[columns]]
    X_train, X_test, y_train, y_test = train_test_split(X_col, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=0.01, max_iter=10000)
    model.fit(X_train, y_train)


    score = model.score(X_test, y_test)
    scores[columns] = score

    df_new = pd.DataFrame(scores.items(), columns=['Features', 'R^2'])
    df_new = df_new.sort_values(by='R^2', ascending=False)

    print(df_new)


'''





