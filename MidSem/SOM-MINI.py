import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

num_samples = 2000
input_dim = 3
data = np.random.random((num_samples, input_dim))

width, height = 10, 10
num_iteration = 2000

som = MiniSom(width, height, input_dim, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, num_iteration)

weights = som.get_weights()

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title('SOM')
plt.imshow(weights)
plt.grid(False)
plt.show()
