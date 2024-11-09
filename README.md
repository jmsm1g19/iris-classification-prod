# Iris Classification Project

This project implements a neural network classifier for the Iris dataset with production-ready features including model experimentation, API deployment, and interactive visualization.

## Project Overview

The project consists of three main components:

1. Model development and experimentation
2. Production API deployment
3. Interactive web interface

## Key Features

1. **Custom Neural Network Implementation**

   - Two-layer neural network with ReLU and Softmax activations
   - Implemented from scratch using NumPy
   - Support for different initialization strategies (Random, He, Xavier)
2. **Experimentation Framework**

   - Grid search for hyperparameter optimization
   - Initialization strategy experiments
   - Cross-validation implementation
3. **Production Deployment**

   - FastAPI backend
   - Streamlit web interface
   - Docker containerization

## Experimental Results

### Data Analysis Insights

Based on the exploratory data analysis:

- Clear separation of Iris Setosa from other classes
- Some overlap between Versicolor and Virginica
- Strong correlation between petal length and width
- Normal distribution for most features except petal width

### Model Performance

Best model configuration achieved through grid search:

- Hidden layer size: 3 neurons
- Learning rate: 0.01
- Number of epochs: 2000
- Batch size: 16
- Initialization: He initialization for hidden layer, Xavier for output layer

Performance metrics:

- Training accuracy: ~98%
- Validation accuracy: ~96%
- Test accuracy: ~95%

### Key Findings

1. **Initialization Impact**

   - He initialization significantly improved training stability
   - Random initialization led to slower convergence
2. **Hyperparameter Sensitivity**

   - Learning rate highly influential on model convergence
   - Batch size affected training stability
   - Hidden layer size showed optimal performance around 3 neurons
3. **Feature Importance**

   - Petal measurements more discriminative than sepal measurements
   - Combined features provide best classification results

## Usage

### Local Development

1. Create and activate virtual environment:
   #TODO
2. Set root directory
3. Run experiments
4. Docker Deployment


Access:

* API documentation: [http://localhost:8000/docs](vscode-file://vscode-app/Applications/Visual%20Studio%20Code%20-%20Insiders.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
* Web interface: [http://localhost:8501](vscode-file://vscode-app/Applications/Visual%20Studio%20Code%20-%20Insiders.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
