import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score

# Load data
data = pd.read_csv('/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv')

# Check missing values
print("Missing values:\n", data.isnull().sum())

# Data dimensions
print("Data shape:", data.shape)

# Reordering columns
new_order = ['Efficiency', 'PLocID', 'Snaptime', 'Distance', 'Practice',
             'precipitation', 'Wind', 'Temp', 'H2F', 'PlayerIDLS', 'OP',
             'PDate', 'Hang', 'SnapLocID', 'Turf', 'Game', 'Grass', 'PlayerIDP']

data = data[new_order]
print("Reordered columns:\n", data.columns.to_list())

# Define X and y
X = data.iloc[:, 1:].values  # all columns except target
y = data.iloc[:, 0].values.reshape(-1, 1)  # target column, reshaped for NN

print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize X (important for neural networks)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.final_input)
        return self.predicted_output

    def backward(self, X, y, learning_rate):
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            if epoch % 4000 == 0:
                loss = np.mean((y - output) ** 2)
                avg_pred = np.mean((y - output) **2 )
                label = "efficient" if avg_pred > 0.5 else "Inefficient"
                print(f"Epoch {epoch}, Loss : {loss:.4f}, Prediction Trend: {label}")
           


# Sample XOR test (optional testing)
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])

# Initialize and train network
nn = NeuralNetwork(input_size=17, hidden_size=24, output_size=1)
nn.train(X, y, epochs=20000, learning_rate=0.01)

# Predictions
preds = nn.feedforward(X)
print("Predictions:\n", preds)


##metrics
accuracy = accuracy_score(avg_pred, y) #% total correct predictions