# Implement a neuron using the error correction algorithm

import numpy as np


class Neuron:
    def __init__(self, features: int):
        self.weights = np.random.rand(features)
        self.bias = np.random.rand()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activate(self, x):
        dot_product = np.dot(x, self.weights) + self.bias
        return self.sigmoid(dot_product)

    def train(self, inputs, targets, lr: int = 0.01, num_iters: int = 10000):
        for _ in range(num_iters):
            idx = np.random.randint(len(inputs))
            x, y = inputs[idx], targets[idx]
            prediction = self.activate(x)
            error = y - prediction
            self.weights += lr * error * x
            self.bias += lr * error


neuron = Neuron(3)
x_train = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y_train = np.array([0, 1, 1, 1])

neuron.train(x_train, y_train)

x_test = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
y_test = np.array([0, 1, 1])

for inputs, output in zip(x_test, y_test):
    print(f"Input: {inputs}\n\tPrediction: {neuron.activate(inputs)}\n\tActual: {output}")
