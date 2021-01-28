import numpy as np
from network.sequential import Sequential
from layer.dense import Dense
from dataset import Datagenerator

dg = Datagenerator(runtimes = 6000)
dg.run()

model = Sequential()
model.add_layer(Dense(input_size = 3, layer_size = 40, randomization = True, activation_function = 'ReLu'))
model.add_layer(Dense(input_size = 40, layer_size = 80, randomization = True, activation_function = 'ReLu'))
model.add_layer(Dense(input_size = 80, layer_size = 50, randomization = True, activation_function = 'Sigmoid'))
model.train(input_data = dg.x, output_data = dg.y, loss_function = 'Quadratic_Loss', optimizer = 'Gradient_Descent', learning_rate = 0.01, epochs = 100)

print('Done')
