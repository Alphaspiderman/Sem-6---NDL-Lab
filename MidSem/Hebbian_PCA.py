import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Neuron:
    def __init__(self, features: int, lr: float):
        self.lr = lr
        self.weights = np.random.rand(features)

    def activate(self, inputs: np.ndarray):
        return np.dot(inputs, self.weights)

    def hebbian_learn(self, inputs: np.ndarray, epochs: int = 100):
        for _ in range(epochs):
            for p in inputs:
                self.weights += self.lr * self.activate(p) * p
            norm = np.linalg.norm(self.weights)
            if norm > 0:
                self.weights /= norm


x = np.random.randint(0, 100, 1000)
noise = np.random.normal(2, 50, size=x.shape)
y = 3 * x + 2 + noise
inputs = np.column_stack((x, y))

mean = np.mean(inputs, axis=0)
centered = inputs - mean

neuron = Neuron(2, 0.00001)
neuron.hebbian_learn(centered)
w = neuron.weights * 500

pca = PCA(1)
pca.fit_transform(centered)

plt.scatter(centered[:, 0], centered[:, 1], alpha=0.3, label="Data points")
plt.quiver(0, 0, *w, color="r", label="Hebbian")
plt.quiver(0, 0, *(pca.components_[0] * 500), color="g", label="PC")
plt.legend()
plt.show()
