import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmAlgorithm

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(SVC(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1)
        score = 1 - accuracy.mean()
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

# Load dataset
dataset = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')

# Extract raw features and feature names
X_raw = dataset.drop(columns=['Efficiency', 'Height', 'Weight']).values
feature_names_raw = dataset.drop(columns=['Efficiency', 'Height', 'Weight']).columns.to_list()

# Impute missing values (mean strategy)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X_raw)

# Identify columns kept by imputer (columns without all NaNs)
cols_kept_mask = ~np.isnan(imputer.statistics_)

# Filter feature names to match imputed X columns
feature_names = np.array(feature_names_raw)[cols_kept_mask]

y = dataset['Efficiency'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1234)

# Define problem and optimization task
problem = SVMFeatureSelection(X_train, y_train)
task_instance = Task(problem=problem, max_iters=100)

# Setup and run PSO algorithm
algorithm = ParticleSwarmAlgorithm(population_size=10, seed=1234)
best_features, best_fitness = algorithm.run(task_instance)

selected_features = best_features > 0.5
print('Number of selected features:', selected_features.sum())
print('Selected features:', ', '.join(feature_names[selected_features]))

# Train and evaluate on selected features
model_selected = SVC()
model_selected.fit(X_train[:, selected_features], y_train)
print('Subset accuracy:', model_selected.score(X_test[:, selected_features], y_test))

# Train and evaluate on all features
model_all = SVC()
model_all.fit(X_train, y_train)
print('All Features Accuracy:', model_all.score(X_test, y_test))

