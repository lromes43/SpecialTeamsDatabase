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

#Create Dictionary to keep track of trends
Score_Trends = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
initial_score = model.score(X_test, y_test)
#print(f"RF Score: {score}")

RFCV = float(cross_val_score(RandomForestClassifier(), X, y).mean())
Score_Trends[0] = RFCV



default = cross_val_score(RandomForestClassifier(random_state=42), X, y).mean()
print(f"Default Score: {default}")

#Comparing Number Estimators (trees)
def best_tree (X_train,y_train):
    Scores = []
    for trees in [1,10,50,55,60,100,200]:
        cross_val_score(RandomForestClassifier(n_estimators=trees, random_state=42), X_train, y_train)
        model = RandomForestClassifier(n_estimators=trees, random_state=42)
        mean_score = float(cross_val_score(model,X,y, cv=5 ).mean())
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

OptimumNumTreeAccuracy = float(cross_val_score(RandomForestClassifier(n_estimators=10, random_state=42), X,y, cv=5).mean())
print(f"chicken: {OptimumNumTreeAccuracy}")
Score_Trends[1] = OptimumNumTreeAccuracy


#optimum number of trees is 10, gives 86.59% Accuracy


##Comparing Criterion
def best_criterion(X_train, y_train):
    criterion = []
    for crit in ['gini', 'entropy']:
       model = RandomForestClassifier(n_estimators=10, criterion = crit, random_state=42)
       mean_score = float(cross_val_score(model, X_train, y_train, cv=5).mean())
       criterion.append((crit, mean_score))
    return criterion
    
Criterion_Scores = best_criterion(X_test, y_test)
print(Criterion_Scores)

OptimumCriterionAccuracy = float(cross_val_score(RandomForestClassifier(n_estimators=10, criterion='gini', random_state=42), X,y, cv=5).mean())
Score_Trends[2] = OptimumCriterionAccuracy

#both Criterion give same score

##Comparing Min Samples Split
def min_split (X_train, y_train):
    split = []
    for s in range(2,50):
        model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split= s, random_state= 42)
        mean_score = float(cross_val_score(model, X_train, y_train, cv=5).mean())
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

OptimumNumSplit = float(cross_val_score(RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=5, random_state=42), X,y).mean())
Score_Trends[3] = OptimumNumSplit

model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=5, random_state=42)
updated_score = float(cross_val_score(model, X_train, y_train, cv=5).mean())
print(f"Updated Score After Estimator, Criterion, Min to Split Tuning: {updated_score}")

model_difference = (updated_score - default) *100
print(model_difference)


##Comparing min Samples Leaf
def min_samples(X,y):
    samples = []
    for samp in range(1, 100):
        model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=5, random_state=42, min_samples_leaf= samp)
        mean_score = float(cross_val_score(model, X,y, cv=5).mean())
        samples.append((samp, mean_score))
    return samples

MinSamples = min_samples(X,y)
print(min_samples)

MinSamplesDF = pd.DataFrame(MinSamples, columns=['NumberSamples', 'Score'])
MinSamplesDF = MinSamplesDF.sort_values(by='Score', ascending=False, ignore_index=True)
print(MinSamplesDF.head())

plt.plot(MinSamplesDF['NumberSamples'], MinSamplesDF['Score'], color = 'purple')
plt.xlabel('# Min Samples') 
plt.ylabel('Score')
plt.title('Min Samples vs Score')
#plt.show()

#Optimum Number of Samples for leaf is 3
OptimumNumLeaf = float(cross_val_score(RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=5, min_samples_leaf=3, random_state=42), X,y).mean())
Score_Trends[4] = OptimumNumLeaf




model = RandomForestClassifier(n_estimators=10, criterion='gini',min_samples_split=5, min_samples_leaf=3, random_state=42)
new_score = float(cross_val_score(model, X_test, y_test).mean())
print(f"Updated Score: {new_score}")

print(Score_Trends)

##Plotting tuning Over time
plt.clf()

data_dict = {"X_Data": [0,1,2,3,4], "y_Data": [0.8942107643600181, '0.8881049298959747', '0.8881049298959747', '0.8972862957937584', '0.9001809136137494']}
print(data_dict)
plt.plot("X_Data", "y_Data", data=data_dict)
plt.xlabel("Number of Parameters Tuned")
plt.ylabel("Accuracy")
plt.title("Number of Parameters Tuned vs Accuracy")
plt.subplots_adjust(left=0.3)
plt.savefig('NumberParametersTunedVSAccuracy')
plt.show()

