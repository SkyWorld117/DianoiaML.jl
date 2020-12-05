import numpy as np
import gc

from network.sequential import Sequential
from layer.dense import Dense
from dataset import Datagenerator

from activation_function.sigmoid import Sigmoid
from activation_function.relu import ReLU
from activation_function.softmax import Softmax

from loss_function.quadratic_loss import Quadratic_Loss
from loss_function.cross_entropy_loss import Cross_Entropy_Loss

from optimizer.gradient_descent import Gradient_Descent
from optimizer.genetic_algorithm import Genetic_Algorithm

from tools.model_management import save_model, load_model

dg = Datagenerator(runtimes = 100)
dg.run()

model = Sequential()
model.add_layer(Dense(input_size = 1, layer_size = 32, randomization = True, activation_function = ReLU))
model.add_layer(Dense(input_size = 32, layer_size = 64, randomization = True, activation_function = ReLU))
model.add_layer(Dense(input_size = 64, layer_size = 32, randomization = True, activation_function = ReLU))
model.add_layer(Dense(input_size = 32, layer_size = 1, randomization = True, activation_function = Sigmoid))

model = Gradient_Descent(sequential=model, input_data=dg.x, output_data=dg.y, loss_function=Quadratic_Loss, learning_rate=0.02, epochs=10, output=True)
#model = Genetic_Algorithm(sequential=model, input_data=dg.x, output_data=dg.y, loss_function=Quadratic_Loss, mutation_rate=0.01, adaptive_mutation_rate=True, epigenetics=True, population=50, epochs=6000, output=True)

save_model(model, 'model0.pkl')
del model
model = load_model('model0.pkl')

print('Done')
