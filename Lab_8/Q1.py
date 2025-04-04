import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)


def train_model(optimizer_name):
    model = models.Sequential()
    model.add(layers.Dense(128, activation="relu", input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=optimizer_name,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
    _, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy with {optimizer_name}: {test_acc:.4f}")


for optimizer in ["Adagrad", "RMSprop", "Adam"]:
    print(f"Training with {optimizer} optimizer:")
    train_model(optimizer)
