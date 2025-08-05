import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.columns.to_list())

X = df.drop(columns=['Efficiency', 'PDate', 'Height', 'Weight'])
print(X.columns.to_list())
y = df.Efficiency
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
#print(f"RF Score: {score}")



default = cross_val_score(RandomForestClassifier(random_state=42), X, y).mean()
print(f"Default Score: {default}")

#Comparing Number Estimators (trees)
def best_tree (X_train,y_train):
    Scores = []
    for trees in [1,10,50,55,60,100,200]:
        cross_val_score(RandomForestClassifier(n_estimators=trees, random_state=42), X_train, y_train)
        model = RandomForestClassifier(n_estimators=trees, random_state=42)
        mean_score = float(cross_val_score(model,X_train,y_train, ).mean())
        Scores.append((trees,mean_score))
    return Scores

Scores = best_tree(X_test,y_test)
print(Scores)

#Convert the Tree list to df for plotting

TreeDF = pd.DataFrame(Scores, columns=['# Trees', 'Score'])
print(TreeDF)

plt.plot(TreeDF['# Trees'], TreeDF['Score'])
plt.xlabel('# trees')
plt.ylabel('Score')
#plt.show()

#optimum number of trees is 10, gives 86.59% Accuracy


##Comparing Criterion
def best_criterion(X_train, y_train):
    criterion = []
    for crit in ['gini', 'entropy']:
       model = RandomForestClassifier(n_estimators=10, criterion = crit, random_state=42)
       mean_score = float(cross_val_score(model, X_train, y_train).mean())
       criterion.append((crit, mean_score))
    return criterion
    
Criterion_Scores = best_criterion(X_test, y_test)
print(Criterion_Scores)

#both Criterion give same score

##Comparing Min Samples Split
def min_split (X_train, y_train):
    split = []
    for s in range(2,50):
        model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split= s, random_state= 42)
        mean_score = float(cross_val_score(model, X_train, y_train).mean())
        split.append((s, mean_score))
    return split

Split_Scores = min_split(X_test, y_test)
print(Split_Scores)

SplitDF = pd.DataFrame(Split_Scores, columns=['MintoSplit', 'Score'])
print(SplitDF)

plt.plot(SplitDF['MintoSplit'], SplitDF['Score'])
plt.xlabel('Min to split')
plt.ylabel('score')
plt.title('Min to Split vs Score')
#plt.show()

model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=5, random_state=42)
updated_score = float(cross_val_score(model, X_train, y_train).mean())
print(f"Updated Score After Estimator, Criterion, Min to Split Tuning: {updated_score}")

model_difference = (updated_score - default) *100
print(model_difference)


##Comparing min Samples Leaf
def min_samples(X_train, y_train):
    samples = []
    for samp in range(1, 100):
        model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=5, random_state=42, min_samples_leaf= samp)
        mean_score = float(cross_val_score(model, X_train, y_train).mean())
        samples.append((samp, mean_score))
    return samples

MinSamples = min_samples(X_test, y_test)
print(min_samples)

MinSamplesDF = pd.DataFrame(MinSamples, columns=['NumberSamples', 'Score'])
plt.plot(MinSamplesDF['NumberSamples'], MinSamplesDF['Score'])
plt.xlabel('# Min Samples') 
plt.ylabel('Score')
plt.title('Min Samples vs Score')
#plt.show()

#No difference

model = RandomForestClassifier(n_estimators=10, criterion='gini',min_samples_split=5, min_samples_leaf=1, random_state=42)
new_score = float(cross_val_score(model, X_test, y_test).mean())
print(f"Updated Score: {new_score}")