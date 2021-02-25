import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential()
model.add(keras.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="sigmoid"))

model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(np.expand_dims(x_train, -1), keras.utils.to_categorical(y_train, 10), batch_size=128, epochs=15)
