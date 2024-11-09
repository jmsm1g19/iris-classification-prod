import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neural network with given sizes."""
        self.parameters = self.initialize_parameters(input_size, hidden_size, output_size)

    def initialize_parameters(self, input_size, hidden_size, output_size):
        """Initialize weights and biases using He and Xavier initialization."""
        W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        b1 = np.zeros((hidden_size, 1))
        W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)
        b2 = np.zeros((output_size, 1))
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def softmax(self, Z):
        """Softmax activation function."""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_propagation(self, X):
        """Perform forward propagation."""
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2, cache

    def compute_loss(self, Y, Y_hat):
        """Compute the categorical cross-entropy loss."""
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-9)) / m
        return loss

    def backpropagation(self, X, Y, cache):
        """Perform backpropagation and compute gradients."""
        m = X.shape[1]
        W2 = self.parameters['W2']

        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2 - Y  # Shape: (output_size, m)
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (A1 > 0)  # Derivative of ReLU
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return gradients

    def update_parameters(self, gradients, learning_rate):
        """Update parameters using gradient descent."""
        self.parameters['W1'] -= learning_rate * gradients['dW1']
        self.parameters['b1'] -= learning_rate * gradients['db1']
        self.parameters['W2'] -= learning_rate * gradients['dW2']
        self.parameters['b2'] -= learning_rate * gradients['db2']

    def predict(self, X):
        """Make predictions with the trained neural network."""
        A2, _ = self.forward_propagation(X)
        predictions = np.argmax(A2, axis=0)
        return predictions
