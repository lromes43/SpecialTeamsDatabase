##Neural Net New


import pandas as pd
import numpy as np
import random 
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import matplotlib
from matplotlib import pyplot as plt

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
print("New Data shape:", data.shape)


X = data.iloc[:, 1:].values  # all columns except target
y_raw = data.iloc[:, 0].values.astype(int)        # shape (331,)
y = np.zeros((y_raw.shape[0], 2))                 # shape (331, 2)
y[np.arange(y_raw.shape[0]), y_raw] = 1


print("X shape:", X.shape)
print("y shape:", y.shape)

##Data input shape: 331 x 17
#Labels, final shape = 331 x 2



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.15, random_state= 42, shuffle= True
)

print("train set: ", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)


##Defing Weights and Bias
B1 = np.zeros((1,24))
W1 = np.random.rand(17,24)

B2 = np.zeros((1,16))
W2 = np.random.rand(24,16)

B3 = np.zeros((1,8))
W3 = np.random.rand(16,8)

B4 = np.zeros((1,4))
W4 = np.random.rand(8,4)

B5 = np.zeros((1,2))
W5 = np.random.rand(4,2)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def softmax(A):
    exp_a = np.exp(A - np.max(A, axis=1, keepdims=True))
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.sum(targets * np.log(predictions)) / targets.shape[0]

learning_rate = 0.01
epochs = 5000

start = time.process_time()

for epoch in range(epochs):
    # Forward pass
    A1 = X_train @ W1 + B1
    S1 = sigmoid(A1)

    A2 = S1 @ W2 + B2
    S2 = sigmoid(A2)

    A3 = S2 @ W3 + B3
    S3 = sigmoid(A3)

    A4 = S3 @ W4 + B4
    S4 = sigmoid(A4)

    A5 = S4 @ W5 + B5
    S5 = softmax(A5)  # Output probabilities

    # Loss calculation
    loss = cross_entropy_loss(S5, y_train)

    # Backpropagation
    output_delta = S5 - y_train  # shape: (samples, 2)

    delta4 = (output_delta @ W5.T) * (S4 * (1 - S4))
    delta3 = (delta4 @ W4.T) * (S3 * (1 - S3))
    delta2 = (delta3 @ W3.T) * (S2 * (1 - S2))
    delta1 = (delta2 @ W2.T) * (S1 * (1 - S1))

    # Update weights and biases
    W5 -= learning_rate * S4.T @ output_delta
    B5 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    W4 -= learning_rate * S3.T @ delta4
    B4 -= learning_rate * np.sum(delta4, axis=0, keepdims=True)

    W3 -= learning_rate * S2.T @ delta3
    B3 -= learning_rate * np.sum(delta3, axis=0, keepdims=True)

    W2 -= learning_rate * S1.T @ delta2
    B2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True)

    W1 -= learning_rate * X_train.T @ delta1
    B1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True)

    # Print loss every 500 epochs
    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

##Testing


# Forward pass on test data
A1_test = X_test @ W1 + B1
S1_test = 1 / (1 + np.exp(-A1_test))

A2_test = S1_test @ W2 + B2
S2_test = 1 / (1 + np.exp(-A2_test))

A3_test = S2_test @ W3 + B3
S3_test = 1 / (1 + np.exp(-A3_test))

A4_test = S3_test @ W4 + B4
S4_test = 1 / (1 + np.exp(-A4_test))

A5_test = S4_test @ W5 + B5
S5_test = softmax(A5_test)

end = time.process_time() # time after
CPU = end - start

# Calculate test loss
test_loss = cross_entropy_loss(S5_test, y_test)
print("Test loss:", test_loss)

# Calculate accuracy on test set
import matplotlib.pyplot as plt  # Make sure this is imported

# Predictions
y_pred = np.argmax(S5_test, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy
accuracy = np.mean(y_pred == y_true)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Log Loss
LogLoss = log_loss(y_test, S5_test)
print("Log Loss:", LogLoss)

# ROC AUC
roc_auc = roc_auc_score(y_test[:, 1], S5_test[:, 1])
print(f"ROC AUC: {roc_auc:.4f}")

# Cross-validation - you can't use cross_val_score directly with your NN
# Here's a placeholder print statement:
print("Cross-validation with custom NN requires manual implementation or sklearn model.")
print(f"CPU Time: {CPU: .4f}")




