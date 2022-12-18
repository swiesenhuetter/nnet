import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output data
y = np.array([[0], [1], [1], [0]])

# Seed the random number generator to get reproducible results
np.random.seed(1)

# Initialize weights randomly with mean 0
W1 = 2 * np.random.random((3, 4)) - 1
W2 = 2 * np.random.random((4, 1)) - 1

# Set the learning rate
learning_rate = 0.1

# Perform backpropagation
for iteration in range(10000):
    # Forward propagation
    layer1 = X
    layer2 = sigmoid(np.dot(layer1, W1))
    layer3 = sigmoid(np.dot(layer2, W2))

    # Calculate the error
    error = y - layer3
    error_delta = error * sigmoid_derivative(layer3)

    # Calculate the error for layer 2
    layer2_error = error_delta.dot(W2.T)
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    # Update the weights
    W2 += layer2.T.dot(error_delta) * learning_rate
    W1 += layer1.T.dot(layer2_delta) * learning_rate

print(f"Final output after training: {layer3}")
