import numpy as np
import matplotlib.pyplot as plt

# Define inputs, outputs, and Gaussian function
inputs = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
outputs = np.array([0, 0, 1, 1])


def gaussian(vector, weights):
    return np.exp(-np.linalg.norm(vector - weights) ** 2)


# Define weights
w1, w2 = np.array([1, 1]), np.array([0, 0])

# Compute Gaussian values
f1 = [gaussian(i, w1) for i in inputs]
f2 = [gaussian(i, w2) for i in inputs]

# Plot results
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(f1[:2], f2[:2], marker="x", label="Class 0")
ax.scatter(f1[2:], f2[2:], marker="o", label="Class 1")
plt.xlabel("Hidden Function 1")
plt.ylabel("Hidden Function 2")
x = np.linspace(0, 1, 10)
y = -x + 1
ax.plot(x, y, label="y = -x + 1")
ax.set_xlim(left=-0.1)
ax.set_ylim(bottom=-0.1)
ax.legend()
plt.show()
