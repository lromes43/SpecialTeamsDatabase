
##Installing Packages

import pandas as pd
import sklearn 
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.svm import SVC



df = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')

print(df.head())

#print(df.info)

#print(df.describe())

