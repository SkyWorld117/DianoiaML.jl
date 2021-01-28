import time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

start = time.time()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(1, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(1, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="softmax"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 128
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

print(time.time()-start)
