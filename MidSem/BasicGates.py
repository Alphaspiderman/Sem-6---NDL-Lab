import numpy as np


class Neuron:
    def __init__(self, features: int, lr: float):
        self.weights = np.random.rand(features)
        self.bias = np.random.rand()
        self.lr = lr

    def activate(self, inputs):
        return np.where(inputs >= 0, 1, 0)

    def train(self, inputs, targets, iters: int = 1000):
        targets = np.array(targets)
        for _ in range(iters):
            linear_output = np.dot(inputs, self.weights) + self.bias
            output = self.activate(linear_output)
            errors = targets - output
            self.weights += self.lr * np.dot(errors, inputs)
            self.bias += self.lr * np.sum(errors)


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "NOR": [1, 0, 0, 0]}
for name, target in targets.items():
    neuron = Neuron(2, 0.1)
    neuron.train(inputs, target)
    print(name)
    output = neuron.activate(np.dot(neuron.weights, inputs.T) + neuron.bias)
    for val, out in zip(inputs, output):
        print(f"Input: {val}: Output: {out}")
    print("")
