# The sequence of backpropagation is Next_Layer to Current_Layer to Last_Layer.

import numpy as np
import multiprocessing
from activation_function.relu import Vectorized_ReLu_Derivative
from activation_function.logistic import Vectorized_Logistic_Derivative

class Gradient_Descent(multiprocessing.Process):

    def __init__(self, id, start_position, learning_rate, batch_size, Current_Layer, Last_Layer, Next_Layer):
        super(Gradient_Descent, self).__init__()
        self.id = id
        self.start_position = start_position
        self.stop_position = self.start_position + batch_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Current_Layer = Current_Layer
        self.Last_Layer = Last_Layer
        self.Next_Layer = Next_Layer

    def run(self):
        if self.Current_Layer.activation_function == 'ReLu':
            derivative = Vectorized_ReLu_Derivative(self.Current_Layer.value)
        elif self.Current_Layer.activation_function == 'Logistic':
            derivative = Vectorized_Logistic_Derivative(self.Current_Layer.value)
        '''
        print('Current_Layer.propagation_units', np.shape(self.Current_Layer.propagation_units))
        print('Current_Layer.weights', np.shape(self.Current_Layer.weights))
        print('derivative', np.shape(derivative))
        print('Next_Layer.propagation_units', np.shape(self.Next_Layer.propagation_units))
        print('Last_Layer.output', np.shape(self.Last_Layer.output))
        '''
        for j in range(self.Current_Layer.layer_size):
            for b in range(self.start_position, self.stop_position):
                for i in range(self.Current_Layer.input_size):
                    self.Current_Layer.propagation_units[i][j][b] = self.Current_Layer.weights[i][j]*derivative[b][j]*self.Next_Layer.propagation_units[j][b]
                    self.Current_Layer.weights[i][j] = self.Current_Layer.weights[i][j] - (1/self.batch_size)*self.learning_rate*self.Next_Layer.propagation_units[j][b]*self.Last_Layer.output[b][i]*derivative[b][j]
                self.Current_Layer.bias[j] = self.Current_Layer.bias[j] - (1/self.batch_size)*self.learning_rate*self.Next_Layer.propagation_units[j][b]*derivative[b][j]
        self.Current_Layer.propagation_units = np.sum(self.Current_Layer.propagation_units, axis = 1)

        global Q
        Q.put((self.Current_Layer.propagation_units, self.Current_Layer.weights, self.Current_Layer.bias))

def Multiprocessing_Gradient_Descent(learning_rate, batch_size, Current_Layer, Last_Layer, Next_Layer):
    Q = multiprocessing.Queue()
    processes = []
    threads = multiprocessing.cpu_count()
    a = batch_size//threads
    b = batch_size%threads
    start_position = 0
    for i in range(threads):
        if b>0:
            processes.append(Gradient_Descent(i, start_position, learning_rate, a+1, Current_Layer, Last_Layer, Next_Layer))
            b = b-1
            start_position = start_position+a+1
        else:
            processes.append(Gradient_Descent(i, start_position, learning_rate, a, Current_Layer, Last_Layer, Next_Layer))
            start_position = start_position+a
        processes[i].start()
    for i in range(threads):
        processes[i].join()

    propagation_units = np.zeros((Current_Layer.input_size, Current_Layer.batch_size))
    weights = np.zeros((Current_Layer.input_size, Current_Layer.layer_size))
    bias = np.zeros((Current_Layer.layer_size, ))
    for i in processes:
        info = Q.get()
        propagation_units = propagation_units+info[0]
        weights = weights+info[1]
        bias = bias+info[2]
    return propagation_units, weights/threads, bias/threads
