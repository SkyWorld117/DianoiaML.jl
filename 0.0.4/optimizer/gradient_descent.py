# The sequence of backpropagation is Next_Layer to Current_Layer to Last_Layer.

import numpy as np
import multiprocessing
from tools.vectorize import Vectorize
from activation_function.relu import ReLu_Derivative
from activation_function.logistic import Logistic_Derivative

def Gradient_Descent(Q, id, start_position, learning_rate, batch_size, derivative, Current_Layer, Last_Layer, Next_Layer):
    stop_position = start_position+batch_size

    for j in range(Current_Layer.layer_size):
        for b in range(start_position, stop_position):
            for i in range(Current_Layer.input_size):
                Current_Layer.propagation_units[i][j][b] = Current_Layer.weights[i][j]*derivative[b][j]*Next_Layer.propagation_units[j][b]
                Current_Layer.weights[i][j] = Current_Layer.weights[i][j]-(1/batch_size)*learning_rate*Next_Layer.propagation_units[j][b]*Last_Layer.output[b][i]*derivative[b][j]
            Current_Layer.bias[j] = Current_Layer.bias[j]-(1/batch_size)*learning_rate*Next_Layer.propagation_units[j][b]*derivative[b][j]
    Current_Layer.propagation_units = np.sum(Current_Layer.propagation_units, axis = 1)

    Q.put((Current_Layer.propagation_units, Current_Layer.weights, Current_Layer.bias))

def Multiprocessing_Gradient_Descent(learning_rate, batch_size, Current_Layer, Last_Layer, Next_Layer):
    if Current_Layer.activation_function == 'ReLu':
        derivative = Vectorize(ReLu_Derivative, 'Activation_Function_Derivative', matrix_a=Current_Layer.value)
    elif Current_Layer.activation_function == 'Logistic':
        derivative = Vectorize(Logistic_Derivative, 'Activation_Function_Derivative', matrix_a=Current_Layer.value)

    Q = multiprocessing.Queue()
    processes = []
    threads = multiprocessing.cpu_count()
    a = batch_size//threads
    b = batch_size%threads
    start_position = 0
    for i in range(threads):
        if b>0:
            processes.append(multiprocessing.Process(target=Gradient_Descent, args=(Q, i, start_position, learning_rate, a+1, derivative, Current_Layer, Last_Layer, Next_Layer)))
            b = b-1
            start_position = start_position+a+1
        else:
            processes.append(multiprocessing.Process(target=Gradient_Descent, args=(Q, i, start_position, learning_rate, a, derivative, Current_Layer, Last_Layer, Next_Layer)))
            start_position = start_position+a
        processes[i].start()

    propagation_units = np.zeros((Current_Layer.input_size, batch_size))
    weights = np.zeros((Current_Layer.input_size, Current_Layer.layer_size))
    bias = np.zeros((Current_Layer.layer_size, ))
    for i in range(threads):
        info = Q.get()
        propagation_units = propagation_units+info[0]
        weights = weights+info[1]
        bias = bias+info[2]

    for i in range(threads):
        processes[i].join()
    return propagation_units, weights/threads, bias/threads
