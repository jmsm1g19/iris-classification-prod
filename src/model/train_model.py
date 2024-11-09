# train_model.py

import os
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from model import NeuralNetwork

# Load environment variables
load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
if not project_root:
    raise ValueError("PROJECT_ROOT is not set. Please run set_project_root.py.")

# Define paths
processed_data_dir = os.path.join(project_root, 'data', 'processed')
model_dir = os.path.join(project_root, 'models')
os.makedirs(model_dir, exist_ok=True)

# Load processed data
processed_data_path = os.path.join(processed_data_dir, 'processed_iris.csv')
data = pd.read_csv(processed_data_path)

# Separate features and target
X = data.drop('class', axis=1).values
y = data['class'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
n_classes = len(np.unique(y))
y_train_one_hot = np.eye(n_classes)[y_train].T  # Shape: (n_classes, m_train)
y_test_one_hot = np.eye(n_classes)[y_test].T    # Shape: (n_classes, m_test)

# Transpose feature matrices
X_train_T = X_train.T  # Shape: (n_features, m_train)
X_test_T = X_test.T    # Shape: (n_features, m_test)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 8  # You can adjust this value
output_size = n_classes
learning_rate = 0.01
n_epochs = 1000
batch_size = 32

# Initialize the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training loop
for epoch in range(n_epochs):
    # Shuffle training data
    permutation = np.random.permutation(X_train_T.shape[1])
    X_shuffled = X_train_T[:, permutation]
    y_shuffled = y_train_one_hot[:, permutation]

    # Mini-batch gradient descent
    for i in range(0, X_train_T.shape[1], batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        y_batch = y_shuffled[:, i:i+batch_size]

        # Forward propagation
        A2, cache = nn.forward_propagation(X_batch)

        # Compute loss
        loss = nn.compute_loss(y_batch, A2)

        # Backward propagation
        gradients = nn.backpropagation(X_batch, y_batch, cache)

        # Update parameters
        nn.update_parameters(gradients, learning_rate)

    # Optionally, print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        # Evaluate on training data
        train_predictions = nn.predict(X_train_T)
        train_accuracy = np.mean(train_predictions == y_train)
        # Evaluate on test data
        test_predictions = nn.predict(X_test_T)
        test_accuracy = np.mean(test_predictions == y_test)
        print(f"Epoch {epoch+1}/{n_epochs} - Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

# Save the trained model
model_path = os.path.join(model_dir, 'model.pkl')
joblib.dump(nn, model_path)
print(f"Model saved to {model_path}")
