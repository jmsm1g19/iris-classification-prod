import numpy as np
root_dir = './'

def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)

def softmax(Z):
    """Softmax activation function."""
    expZ = np.exp(Z - np.max(Z, axis=0))
    return expZ / expZ.sum(axis=0, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    """Initialize weights and biases for a neural network with 1 hidden layer."""
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def initialize_parameters_he_xavier(input_size, hidden_size, output_size):
    """Initialize weights with He initialization for ReLU and Xavier for softmax."""
    # He initialization for ReLU hidden layer
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((hidden_size, 1))
    # Xavier/Glorot initialization for the output layer (before softmax)
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, parameters):
    """Perform forward propagation."""
    W1, b1, W2, b2 = parameters.values()
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_loss(Y, Y_hat):
    """
    Compute the Categorical Cross-Entropy loss.
    
    Parameters:
    - Y: true "one-hot" encoded labels of shape [n_classes, n_samples]
    - Y_hat: predicted probabilities of shape [n_classes, n_samples]
    
    Returns:
    - loss: The average Categorical Cross-Entropy loss across all samples
    """
    m = Y.shape[1]  # Number of samples
    loss = -np.sum(Y * np.log(Y_hat + 1e-9)) / m
    return loss

def backpropagation(X, Y, cache, parameters):
    """Perform backpropagation with detailed explanation on the derivation of dZ2."""
    m = X.shape[1]
    Z1, A1, Z2, A2 = cache.values()
    W2 = parameters["W2"]
    
    # Start with the derivative of the cross-entropy loss with respect to A2 (softmax outputs)
    # In practice, this calculation simplifies to A2 - Y, but let's break it down:
    # For cross-entropy loss L = -sum(Y * log(A2))
    # dL/dA2 for each class i = -Y/A2
    # For softmax function A2_i = exp(Z2_i) / sum(exp(Z2_j))
    # The derivative of softmax output A2_i w.r.t Z2_i includes two cases:
    # i) when i = j (diagonal elements), ii) when i != j (off-diagonal elements)
    # Combining these gives the simplified form dZ2 = A2 - Y, directly applicable for gradient updates
    
    dZ2 = A2 - Y  # Direct result of combined derivative calculation
    
    # Gradient of W2
    dW2 = np.dot(dZ2, A1.T) / m
    # Gradient of b2
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    # Backpropagate to the first layer
    dA1 = np.dot(W2.T, dZ2)
    # Derivative of ReLU
    dZ1 = dA1 * (A1 > 0)
    # Gradient of W1
    dW1 = np.dot(dZ1, X.T) / m
    # Gradient of b1
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

def update_parameters(parameters, gradients, learning_rate=0.01):
    """Update parameters using gradient descent."""
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters

# %%
def predict(X, parameters):
    """
    Make a prediction with the neural network.
    
    Parameters:
    - X: input data of shape (n_features, m_samples)
    - parameters: the weights and biases of the neural network
    
    Returns:
    - predictions: The predictions of the network
    """
    A2, _ = forward_propagation(X, parameters)
    predictions = np.argmax(A2, axis=0)
    return predictions