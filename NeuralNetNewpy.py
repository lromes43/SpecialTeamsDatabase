##Neural Net New


import pandas as pd
import numpy as np
import random 

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



##Defining Bias
##Initalize to 0

B1 = np.zeros((1,24))


##inital weights
W1 = np.random.rand(17,24)




##training occurs within 2 loops, inner loop iteratres through all input label pairs, outer specifies how ofgten iterate through every


epochs = 4
for epoch in range(epochs):
    print(f"Epoch {epoch +1 } / {epoch}")
    for i in range(X.shape[0]):
        x_sample = X[i].reshape(-1,1)
        y_sample = y[i].reshape(-1,1)

        ##print(f"Sample {i + 1}: x shape = {x_sample.shape}, y shape = {y_sample.shape}")


##Forward Propogration, used to transform input values to output values


Z1 = X @ W1 + B1
A1 = 1 / (1 + np.exp(-Z1)) ##Sigmoid Function


###Second Layer

#Second Bias 

B2 = np.zeros((1,16))

##second weights
W2 = np.random.rand(24,16)

Z2 = Z1 @ W2 + B2
A2 = 1 / (1+np.exp(-Z2))

###Third Layer
#Bias
B3 = np.zeros((1,8))

##Third Weights

W3 = np.random.rand(16,8)

Z3 = Z2 @ W3 + B3
A3 = 1 / (1 + np.exp(-Z3))


##Fourth Layer

#Bias
B4 = np.zeros((1,4))
#fourth weights
W4 = np.random.rand(8,4)

Z4 = Z3 @ W4 + B4
A4 = 1 / (1 + np.exp(-Z4))

##Comparing Outputs to Labels


##Fifth Layer
B5 = np.zeros((1,2))
W5 = np.random.rand(4,2)
Z5 = Z4 @ W5 + B5
#Use softmax instead of sigmoid

def softmax (z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

A5 = softmax(Z5)



##Backpropogation

output_error = y - A5
output_delta = output_error * A5 * (1 - A5)
hidden_error = output_delta @ W5.T
hidden_delta = hidden_error * A4 * (1-A4)
hidden_error3 = hidden_delta @ W4.T
hidden_delta3 = hidden_error3 * A3 * (1-A3)
hidden_error2 = hidden_error3 @ W3.T
hidden_delta2 = hidden_error2 * A2 * (1-A2)
hidden_error1 = hidden_delta2 @ W2.T
hidden_delta1 = hidden_error1 * A1 * (1-A1)

learning_rate = 0.001

W5 -= learning_rate * (A4.T @ output_delta)
B5 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

W4 -= learning_rate * (A3.T @ hidden_delta)
B4 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

W3 -= learning_rate * (A2.T @ hidden_delta3)
B3 -= learning_rate * np.sum(hidden_delta3, axis=0, keepdims=True)

W2 -= learning_rate * (A1.T @ hidden_delta2)
B2 -= learning_rate * np.sum(hidden_delta2, axis=0, keepdims=True)

W1 -= learning_rate * (X.T @ hidden_delta1)
B1 -= learning_rate * np.sum(hidden_delta1, axis=0, keepdims=True)




