# Two-Layer Neural Network on MNIST Dataset

### Pritpal Singh

This repository contains an instructional implementation of a simple two-layer neural network, trained on the MNIST digit recognizer dataset. The goal of this project is to provide a clear and concise example to help you understand the underlying mathematics of neural networks.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Neural Network Architecture](#neural-network-architecture)
- [Implementation](#implementation)
  - [Forward Propagation](#forward-propagation)
  - [Backward Propagation](#backward-propagation)
  - [Parameter Updates](#parameter-updates)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates a basic implementation of a neural network with one hidden layer, designed to classify digits from the MNIST dataset. The notebook walks through the steps of initializing parameters, performing forward and backward propagation, updating parameters, and making predictions.

## Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits, ranging from 0 to 9. Each image is flattened into a 784-element vector. The dataset is split into a training set and a development set.

## Neural Network Architecture

The neural network has the following architecture:

- **Input layer**: 784 units (corresponding to the 784 pixels of each image)
- **Hidden layer**: 10 units with ReLU activation
- **Output layer**: 10 units with softmax activation (corresponding to the 10 digit classes)

## Implementation

The implementation involves several key steps:

### Forward Propagation

The forward propagation equations are:
- \( Z^{[1]} = W^{[1]} X + b^{[1]} \)
- \( A^{[1]} = \text{ReLU}(Z^{[1]}) \)
- \( Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} \)
- \( A^{[2]} = \text{softmax}(Z^{[2]}) \)

### Backward Propagation

The backward propagation equations are:
- \( dZ^{[2]} = A^{[2]} - Y \)
- \( dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T} \)
- \( dB^{[2]} = \frac{1}{m} \sum dZ^{[2]} \)
- \( dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot \text{ReLU}'(Z^{[1]}) \)
- \( dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T} \)
- \( dB^{[1]} = \frac{1}{m} \sum dZ^{[1]} \)

### Parameter Updates

Parameters are updated using gradient descent:
- \( W^{[2]} := W^{[2]} - \alpha dW^{[2]} \)
- \( b^{[2]} := b^{[2]} - \alpha db^{[2]} \)
- \( W^{[1]} := W^{[1]} - \alpha dW^{[1]} \)
- \( b^{[1]} := b^{[1]} - \alpha db^{[1]} \)

## Usage

To use this notebook, follow these steps:

1. **Load the dataset**: The dataset should be available as a CSV file.
2. **Initialize parameters**: Use `init_params` function to initialize weights and biases.
3. **Train the model**: Use `gradient_descent` function to train the neural network.
4. **Make predictions**: Use `make_predictions` function to predict the classes of new data.
5. **Test the model**: Use `test_prediction` function to visualize and test predictions.

### Example

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print(f'Development Set Accuracy: {accuracy * 100}%')
```
![image](https://github.com/pritpalcodes/neural_network_from_scratch/assets/90276050/0289292d-6d23-49d2-97fd-6d48ecb9b8e1)
![image](https://github.com/pritpalcodes/neural_network_from_scratch/assets/90276050/e92669f9-a9c2-4d9e-91d5-b9582a2f606a)

