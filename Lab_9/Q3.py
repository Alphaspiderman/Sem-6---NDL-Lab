import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_cell_forward(x_t, h_prev, c_prev, p):
    concat = np.vstack((h_prev, x_t))

    # Gates and candidate memory
    f_t = sigmoid(p["W_f"] @ concat + p["b_f"])
    i_t = sigmoid(p["W_i"] @ concat + p["b_i"])
    c_tilde = np.tanh(p["W_c"] @ concat + p["b_c"])

    # Update cell state
    c_next = f_t * c_prev + i_t * c_tilde

    # Output gate and next hidden state
    o_t = sigmoid(p["W_o"] @ concat + p["b_o"])
    h_next = o_t * np.tanh(c_next)

    return h_next, c_next


# Random initializations
np.random.seed(2)
inp_s, hid_s = 3, 5
x_t = np.random.randn(inp_s, 1)
h_prev = np.random.randn(hid_s, 1)
c_prev = np.random.randn(hid_s, 1)

# Random weight and bias initialization
parameters = {
    key: (
        np.random.randn(hid_s, hid_s + inp_s)
        if "W" in key
        else np.random.randn(hid_s, 1)
    )
    for key in ["W_f", "b_f", "W_i", "b_i", "W_c", "b_c", "W_o", "b_o"]
}

# Perform a single LSTM step
h_next, c_next = lstm_cell_forward(x_t, h_prev, c_prev, parameters)
print("Next hidden state h_next:\n", h_next)
print("\nNext cell state c_next:\n", c_next)
