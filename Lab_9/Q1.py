import numpy as np

def rnn_cell_forward(x_t, h_prev, W_x, W_h, b):
    return np.tanh(W_x @ x_t + W_h @ h_prev + b)

# Initialize parameters
np.random.seed(1)
inp, hid_s = 3, 5
x_t = np.random.randn(inp, 1)
h_prev = np.random.randn(hid_s, 1)
W_x = np.random.randn(hid_s, inp)
W_h = np.random.randn(hid_s, hid_s)
b = np.random.randn(hid_s, 1)

# Compute the next hidden state
h_t = rnn_cell_forward(x_t, h_prev, W_x, W_h, b)
print("Next hidden state h_t:\n", h_t)
