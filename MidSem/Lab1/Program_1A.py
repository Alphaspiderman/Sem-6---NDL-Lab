import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.maximum(0.1 * x, x)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def binary_step(x):
    return np.where(x < 0, 0, 1)


def linear(x):
    return 4 * x


functions = {"Sigmoid": sigmoid, "tanh": tanh, "ReLU": relu, "Leaky ReLU": leaky_relu, "Softmax": softmax,
             "Binary Step": binary_step, "Linear": linear}

vals = [-5, - 2, 0, 2, 5]
for name, func in functions.items():
    print(f"{name}")
    for val in vals:
        print(f"{val} -> {func(val):.2f}")
    print("\n")
