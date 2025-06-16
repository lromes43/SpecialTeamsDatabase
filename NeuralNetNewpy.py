##Neural Net New


import pandas as pd
import numpy as np
import random 
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split

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
    X, y, test_size= 0.2, random_state= 42, shuffle= True
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

# Calculate test loss
test_loss = cross_entropy_loss(S5_test, y_test)
print("Test loss:", test_loss)

# Calculate accuracy on test set
y_pred = np.argmax(S5_test, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")




'''


##Layer 1
#Bias
B1 = np.zeros((1,24))
#Weights
W1 = np.random.rand(17,24)

##training occurs within 2 loops, inner loop iteratres through all input label pairs, outer specifies how ofgten iterate through every

epochs = 4
for epoch in range(epochs):
    print(f"Epoch {epoch +1 } / {epoch}")
    for i in range(X.shape[0]):
        x_sample = X[i].reshape(-1,1)
        y_sample = y[i].reshape(-1,1)

        ##print(f"Sample {i + 1}: x shape = {x_sample.shape}, y shape = {y_sample.shape}")

#Forward Propogation
A1= X @ W1 + B1
S1 = 1 / (1 + np.exp(-A1)) ##Sigmoid Function

###Second Layer
#Second Bias 
B2 = np.zeros((1,16))
#second weights
W2 = np.random.rand(24,16)
#Forward Propogation
A2 = A1 @ W2 + B2
S2 = 1 / (1+np.exp(-A2))

###Third Layer
#Bias
B3 = np.zeros((1,8))
##Third Weights
W3 = np.random.rand(16,8)
#Forward Propogation
A3 = A2 @ W3 + B3
S3 = 1 / (1 + np.exp(-A3))


##Fourth Layer
#Bias
B4 = np.zeros((1,4))
#fourth weights
W4 = np.random.rand(8,4)
#Forward Propogation
A4 = A3 @ W4 + B4
S4 = 1 / (1 + np.exp(-A4))

##Fifth Layer
B5 = np.zeros((1,2))
W5 = np.random.rand(4,2)
#Forward Propogation
A5 = A4 @ W5 + B5


#Softmax
def softmax (A):
    exp_a = np.exp(A - np.max(A, axis=1, keepdims=True))
    S = exp_a / np.sum(exp_a, axis=1, keepdims= True)
    return S

S5 = softmax(A5)

S1 = softmax(S1)
S2 = softmax(S2)
S3 = softmax(S3)
S4 = softmax(S4)
S5 = softmax(S5)


##Cross Entropy Loss
def Loss1(S,y):
    S = np.clip(S, 1e-15, 1 - 1e-15)
    L = -np.sum(y * np.log((S)) / y.shape[0])
    return L
    
Cross_Entropy_Loss = Loss1(S5, y)
print("Cross Entropy Loss: ", Cross_Entropy_Loss)




##Backpropogation
#Output Unit Error
pred = S5
true = y

output_delta = pred - true
print(output_delta)

##Hidden Unit Error
#Layer 4
delta4 = S4 * (1-S4) * (output_delta @ W5.T)
print("Delta 4 Shape", delta4.shape)

##Layer 3
delta3 = S3 * (1-S3) * (delta4 @ W4.T)
print("Delta 3 Shape", delta3.shape)

##Layer 2 
delta2 = S2 * (1-S2) * (delta3 @ W3.T)
print("Delta 2 Shape", delta2.shape)

##Layer 1
delta1 = S1 * (1-S1) * (delta2 @ W2.T)
print("Delta 1 Shape", delta1.shape)


##Updating Weights
learning_rate = 0.01
W5 -= learning_rate * A4.T @ output_delta
B5 -= learning_rate * np.sum(output_delta, axis=0, keepdims= True)

W4 -= learning_rate * A3.T @ delta4
B4 -= learning_rate * np.sum(delta4,axis=0, keepdims= True)

W3 -= learning_rate * A2.T @ delta3
B3 -= learning_rate * np.sum(delta3, axis=0, keepdims= True)

W2 -= learning_rate * A1.T @ delta2
B2 -= learning_rate * np.sum(delta2, axis= 0, keepdims= True)

W1 -= learning_rate * X.T @ delta1
B1 -= learning_rate * np.sum(delta1, axis= 0, keepdims= True)


print(f"Weights for 5: {W5} \n Bias for 5: {B5}")
print(f"Weights for 4: {W4} \n Bias for 4: {B4}")
print(f"Weights for 3: {W3} \n Bias for 3: {B3}")
print(f"Weights for 2: {W2} \n Bias for 2: {B2}")
print(f"Weights for 1: {W1} \n Bias for 1: {B1}")





##Training Neural Net




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






'''




