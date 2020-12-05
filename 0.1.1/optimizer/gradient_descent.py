# The sequence of backpropagation is Next_Layer to Current_Layer to Last_Layer.

import numpy as np
import multiprocessing, gc
from tools.vectorize import Vectorize
from loss_function.standard_loss import Standard_Loss

def Calculating_Unit(Q, id, start_position, learning_rate, batch_size, Current_Layer, Last_Layer, Next_Layer):
    stop_position = start_position+batch_size

    gradient = np.zeros(np.shape(Current_Layer.value))
    for b in range(start_position, stop_position):
        for j in range(Current_Layer.layer_size):
            if Next_Layer.propagation_units[j][b]!=0:
                gradient[b] += Current_Layer.activation_function.derivative(Current_Layer.value[b], j)*Next_Layer.propagation_units[j][b]

    for b in range(start_position, stop_position):
        for j in range(Current_Layer.layer_size):
            for i in range(Current_Layer.input_size):
                Current_Layer.propagation_units[i][j][b] = Current_Layer.weights[i][j]*gradient[b][j]
                Current_Layer.weights[i][j] -= (1/batch_size)*learning_rate*gradient[b][j]*Last_Layer.output[b][i]
            Current_Layer.biases[j] -= (1/batch_size)*learning_rate*gradient[b][j]
    Current_Layer.propagation_units = np.sum(Current_Layer.propagation_units, axis=1)
    Q.put((Current_Layer.propagation_units, Current_Layer.weights, Current_Layer.biases))

def Multi_Processing(learning_rate, batch_size, Current_Layer, Last_Layer, Next_Layer):
    Q = multiprocessing.Queue()
    processes = []
    threads = multiprocessing.cpu_count()
    a = batch_size//threads
    b = batch_size%threads
    start_position = 0
    for i in range(threads):
        if b>0:
            processes.append(multiprocessing.Process(target=Calculating_Unit, args=(Q, i, start_position, learning_rate, a+1, Current_Layer, Last_Layer, Next_Layer)))
            b = b-1
            start_position = start_position+a+1
        else:
            processes.append(multiprocessing.Process(target=Calculating_Unit, args=(Q, i, start_position, learning_rate, a, Current_Layer, Last_Layer, Next_Layer)))
            start_position = start_position+a
        processes[i].start()

    propagation_units = np.zeros((Current_Layer.input_size, batch_size))
    weights = np.zeros((Current_Layer.input_size, Current_Layer.layer_size))
    biases = np.zeros((Current_Layer.layer_size, ))
    for i in range(threads):
        info = Q.get()
        propagation_units += info[0]
        weights += info[1]
        biases += info[2]

    for i in range(threads):
        processes[i].join()
    return propagation_units, weights/threads, biases/threads

class Hidden_Output_Layer:

    def __init__(self):
        self.propagation_units = []

def Gradient_Descent(sequential=None, input_data=[], output_data=[], loss_function=None, learning_rate=0.01, epochs=6000, output=True):
    batch_size = np.shape(input_data)[0]
    sequential.layers[0].output = input_data
    HOL = Hidden_Output_Layer()
    sequential.layers.append(HOL)

    if output:
        for e in range(epochs):
            print('Epoch', e+1)
            for i in range(1, len(sequential.layers)-1):
                print('\r Forward propagating...Layer'+str(i), end='')
                sequential.layers[i].run(sequential.layers[i-1].output)
            print()
        
            print(' Calculating the loss... ')
            sequential.layers[-1].propagation_units = Vectorize(loss_function.propagation, matrix_a=sequential.layers[-2].output, matrix_b=output_data)

            print('  Current loss =',Vectorize(Standard_Loss.function, matrix_a=sequential.layers[-2].output, matrix_b=output_data))
            #print('  Current accuracy = ', 1-Vectorize(Loss_Function, matrix_a=sequential.layers[-2].output, matrix_b=output_data)/batch_size)

            for i in range(len(sequential.layers)-2, 0, -1):
                print('\r Back propagating...Layer'+str(i), end='')
                sequential.layers[i].propagation_units, sequential.layers[i].weights, sequential.layers[i].biases = Multi_Processing(learning_rate, batch_size, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1])
            print()

    if not output:
        for e in range(epochs):
            for i in range(1, len(sequential.layers)-1):
                sequential.layers[i].run(sequential.layers[i-1].output)
        
            sequential.layers[-1].propagation_units = Vectorize(loss_function.propagation, matrix_a=sequential.layers[-2].output, matrix_b=output_data)

            for i in range(len(sequential.layers)-2, 0, -1):
                sequential.layers[i].propagation_units, sequential.layers[i].weights, sequential.layers[i].biases = Multi_Processing(learning_rate, batch_size, sequential.layers[i], sequential.layers[i-1], sequential.layers[i+1])

    del sequential.layers[-1]
    gc.collect()

    return sequential