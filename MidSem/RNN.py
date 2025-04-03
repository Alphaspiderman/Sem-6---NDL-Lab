import tensorflow as tf
import keras

class SimpleRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = keras.layers.SimpleRNN(hidden_size, return_sequences=False)
        self.fc = keras.layers.Dense(output_size)

    def call(self, x):
        return self.fc(self.rnn(x))


# Training
input_size, hidden_size, output_size = 3, 5, 1
model = SimpleRNN(input_size, hidden_size, output_size)
model.compile(optimizer='adam', loss='mse')

data = tf.random.normal((2, 4, input_size))
targets = tf.random.normal((2, output_size))
model.fit(data, targets, epochs=10, verbose=2)
