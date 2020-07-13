import numpy as np
import gc
from layer.dense import Dense

class Hidden_Input_Layer:

    def __init__(self):
        self.output = []

class Sequential:

    def __init__(self):
        self.layers = []
        HIL = Hidden_Input_Layer()
        self.layers.append(HIL)

    def add_layer(self, layer):
        self.layers.append(layer)

    def run(self, input):
        self.layers[0].output = input
        for i in range(1, len(self.layers)-1):
            self.layers[i].run(self.layers[i-1].output)
        return self.layers[len(self.layers)-2].output
