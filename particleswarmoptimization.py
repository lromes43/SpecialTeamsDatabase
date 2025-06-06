
##Installing Packages

import pandas as pd
import sklearn 
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.svm import SVC

import niapy
from niapy import problems
from niapy import task
from niapy import algorithms
from niapy.algorithms import basic
from niapy.algorithms.basic import ParticleSwarmAlgorithm

##Packages


##Loading Data in
df = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')


##Head of Dataframe
print(df.head()) 


class SVMFeatureSelection (Problem):
    def _init_(self, X_train, y_train, alpha = 0.99):
        super(). _init_ (dimension = X_train.shape[1], lower = 0, upper = 1)
        self.X_train 

