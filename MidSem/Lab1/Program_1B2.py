# Implement a neuron using the memory based learning algorithm

import numpy as np
from collections import Counter


class Neuron:
    def __init__(self, k: int, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = [self._predict(x) for x in x_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.distance(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)
        k_nearest_labels = [self.y_train[i] for i in k_indices[:self.k]]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x2 - x1) ** 2))


x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])
neuron = Neuron(3, x_train, y_train)

x_test = np.array([[5, 6], [0, 1]])
y_pred = neuron.predict(x_test)

for x, y in zip(x_test, y_pred):
    print(f"Input: {x}\n\tPrediction: {y}")
