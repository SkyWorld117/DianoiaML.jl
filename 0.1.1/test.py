import numpy as np
import gc, time

from network.sequential import Sequential
from layer.dense import Dense
from dataset import Datagenerator

from activation_function.sigmoid import Sigmoid
from activation_function.relu import ReLU
from activation_function.softmax import Softmax
from activation_function.tanh import tanh

from loss_function.quadratic_loss import Quadratic_Loss
from loss_function.cross_entropy_loss import Cross_Entropy_Loss

from optimizer.gradient_descent import Gradient_Descent
from optimizer.genetic_algorithm import Genetic_Algorithm

from tools.model_management import save_model, load_model

input_data = np.array([[0, 4, 8, 2],[1, 5, 9, 3],[2, 6, 0, 4],[3, 7, 1, 5]])
output_data = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
Start = time.time()
'''
model = Sequential()
model.add_layer(Dense(input_size = 4, layer_size = 5, randomization = True, activation_function = ReLU))
model.add_layer(Dense(input_size = 5, layer_size = 5, randomization = True, activation_function = tanh))
model.add_layer(Dense(input_size = 5, layer_size = 5, randomization = True, activation_function = Sigmoid))
model.add_layer(Dense(input_size = 5, layer_size = 2, randomization = True, activation_function = Softmax))

model = Gradient_Descent(sequential=model, input_data=input_data, output_data=output_data, loss_function=Cross_Entropy_Loss, learning_rate=0.02, epochs=10, output=True)
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(4,)))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.tanh))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(input_data, output_data, epochs=10)
print(time.time()-Start)
