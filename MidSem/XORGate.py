import tensorflow as tf
import numpy as np
import keras

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])


def create_model(lr):
    model = keras.Sequential([
        keras.layers.Dense(2, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    return model


for lr in [0.5, 0.1, 0.01]:
    print(f"Learning rate: {lr}")
    model = create_model(lr)
    model.fit(inputs, outputs, epochs=500, verbose=0)

    for i in inputs:
        print(f"Input: {i}	Predicted: {model.predict(i.reshape(1, -1))[0][0]:.2f}")
    print("")
