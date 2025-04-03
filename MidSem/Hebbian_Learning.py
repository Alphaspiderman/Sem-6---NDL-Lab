import numpy as np


class Hebbian:
    def __init__(self, features: int, lr: float):
        self.weights = np.random.rand(features)
        self.lr = lr

    def activate(self, inputs):
        return np.dot(self.weights, inputs)

    def train(self, inputs, iters: int = 1000):
        for _ in range(iters):
            idx = np.random.randint(len(inputs))
            val = inputs[idx]
            self.weights += self.lr * self.activate(val) * val


neuron = Hebbian(3, 0.1)
inputs = np.array([[0.5, 0.3, 0.2],
                   [1.0, 0.6, 0.4]])
neuron.train(inputs)
print(neuron.weights)
