import numpy as np

def rnn_forward(X, h0, W_x, W_h, b_h, W_y, b_y):
    steps, hid_s = X.shape[0], h0.shape[0]
    out_s = W_y.shape[0]

    h = np.zeros((steps, hid_s, 1))
    y = np.zeros((steps, out_s, 1))
    h_prev = h0

    for t in range(steps):
        x_t = X[t].reshape(-1, 1)
        h_t = np.tanh(W_x @ x_t + W_h @ h_prev + b_h)
        y_t = W_y @ h_t + b_y

        h[t], y[t], h_prev = h_t, y_t, h_t

    return h, y

# Example dimensions
steps, inp_s, hid_s, out_s = 4, 3, 5, 2

# Random initializations
np.random.seed(1)
X = np.random.randn(steps, inp_s)
h0 = np.zeros((hid_s, 1))
W_x = np.random.randn(hid_s, inp_s)
W_h = np.random.randn(hid_s, hid_s)
b_h = np.random.randn(hid_s, 1)
W_y = np.random.randn(out_s, hid_s)
b_y = np.random.randn(out_s, 1)

# Perform full forward propagation
h, y = rnn_forward(X, h0, W_x, W_h, b_h, W_y, b_y)
print("Hidden states:\n", h)
print("\nOutputs:\n", y)
