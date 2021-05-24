import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="tanh"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="tanh"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(x_train, keras.utils.to_categorical(y_train, 10), batch_size=128, epochs=10)
