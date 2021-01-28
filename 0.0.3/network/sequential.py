import numpy as np
import gc
from layer.dense import Dense
from optimizer.gradient_descent import Multiprocessing_Gradient_Descent
from loss_function.quadratic_loss import Vectorized_Quadratic_Loss_Propagation_Units, Vectorized_Quadratic_Loss

class Hidden_Input_Layer:

    def __init__(self):
        self.output = []

class Hidden_Output_Layer:

    def __init__(self):
        self.propagation_units = []

class Sequential:

    def __init__(self):
        self.layers = []
        HIL = Hidden_Input_Layer()
        self.layers.append(HIL)

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, input_data = [], output_data = [], loss_function = 'Quadratic_Loss', optimizer = 'Gradient_Descent', learning_rate = 0.01, epochs = 6000):
        batch_size = np.shape(input_data)[0]

        self.layers[0].output = input_data
        HOL = Hidden_Output_Layer()
        self.layers.append(HOL)

        for e in range(epochs):
            print('Epoch', e+1)
            for i in range(1, len(self.layers)-1):
                print('\r Forward propagating...Layer'+str(i), end='')
                self.layers[i].run(self.layers[i-1].output)
            print()

            print(' Calculating the loss... ', end='')
            if loss_function == 'Quadratic_Loss':
                self.layers[len(self.layers)-1].propagation_units = Vectorized_Quadratic_Loss_Propagation_Units(self.layers[len(self.layers)-2].output, output_data)
                print('current loss =',np.sum(Vectorized_Quadratic_Loss(self.layers[len(self.layers)-2].output, output_data)))

            for i in range(len(self.layers)-2, 0, -1):
                print(' Back propagating...Layer'+str(i))
                self.layers[i].propagation_units, self.layers[i].weights, self.layers[i].bias = Multiprocessing_Gradient_Descent(learning_rate, batch_size, self.layers[i], self.layers[i-1], self.layers[i+1])
                
            gc.collect()

    def run(self, input):
        self.layers[0].output = input
        for i in range(1, len(self.layers)-1):
            self.layers[i].run(self.layers[i-1].output)
        return self.layers[len(self.layers)-2].output
