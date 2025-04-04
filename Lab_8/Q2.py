import numpy as np
import keras
from keras import datasets, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def le_net():

    model = keras.models.Sequential(
        [
            layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def alexnet():

    model = keras.models.Sequential(
        [
            layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def vgg():

    model = keras.models.Sequential(
        [
            layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def placesnet():

    model = keras.models.Sequential(
        [
            layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


types = {
    "LeNet": le_net(),
    "AlexNet": alexnet(),
    "VGG": vgg(),
    "PlacesNet": placesnet(),
}


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

accs = {}

for key, model in types.items():

    print(f"Training with: {key}")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    accs[key] = history

for name, h in accs.items():
    plt.figure(figsize=(6, 4))
    plt.plot(h.history["accuracy"], label=f"{name} Train Acc")
    plt.plot(h.history["val_accuracy"], label=f"{name} Val Acc", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy and Validation Accuracy for {name}")
    plt.legend()
    plt.show()
