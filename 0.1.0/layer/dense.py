import numpy as np
from tools.vectorize import Vectorize

class Dense:

    def __init__(self, input_size = 0, layer_size = 0, randomization = True, activation_function = None):
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation_function = activation_function
        if randomization:
            self.weights = 0.1*np.random.randn(input_size, layer_size)
            self.biases = 0.1*np.random.randn(layer_size)
        else:
            self.weights = np.zeros((input_size, layer_size))
            self.biases = np.zeros(layer_size)

    def run(self, input):
        self.value = np.matmul(input, self.weights) + self.biases
        self.output = Vectorize(self.activation_function.function, matrix_a=self.value)
        self.propagation_units = np.zeros((self.input_size, self.layer_size, np.shape(input)[0]))
