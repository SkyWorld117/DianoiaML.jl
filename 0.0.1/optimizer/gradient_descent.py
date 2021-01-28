# The sequence of backpropagation is Next_Layer to Current_Layer to Last_Layer.

import numpy as np
from activation_function.relu import ReLu_Derivative
from activation_function.sigmoid import Sigmoid_Derivative

def Gradient_Descent(learning_rate, Current_Layer, Last_Layer, Next_Layer):
    Batch_Size = np.shape(Current_Layer.propagation_units)[2]

    for i in range(np.shape(Current_Layer.propagation_units)[0]):
        for j in range(np.shape(Current_Layer.propagation_units)[1]):
            for b in range(Batch_Size):
                print('\r  Generating propagation units... '+str(i+1)+'/'+str(np.shape(Current_Layer.propagation_units)[0])+' '+str(j+1)+'/'+str(np.shape(Current_Layer.propagation_units)[1])+' '+str(b+1)+'/'+str(Batch_Size), end='')
                if Current_Layer.activation_function == 'ReLu':
                    Current_Layer.propagation_units[i][j][b] = Current_Layer.weights[i][j]*ReLu_Derivative(Current_Layer.value[b][j])*Next_Layer.propagation_units[j][b]
                elif Current_Layer.activation_function == 'Sigmoid':
                    Current_Layer.propagation_units[i][j][b] = Current_Layer.weights[i][j]*Sigmoid_Derivative(Current_Layer.value[b][j])*Next_Layer.propagation_units[j][b]
    Current_Layer.propagation_units = np.sum(Current_Layer.propagation_units, axis = 1)
    print()

    for i in range(np.shape(Current_Layer.weights)[0]):
        for j in range(np.shape(Current_Layer.weights)[1]):
            for b in range(Batch_Size):
                print('\r  Optimizing weights... '+str(i+1)+'/'+str(np.shape(Current_Layer.weights)[0])+' '+str(j+1)+'/'+str(np.shape(Current_Layer.weights)[1])+' '+str(b+1)+'/'+str(Batch_Size), end='')
                if Current_Layer.activation_function == 'ReLu':
                    Current_Layer.weights[i][j] = Current_Layer.weights[i][j] - (1/Batch_Size)*learning_rate*Next_Layer.propagation_units[j][b]*Last_Layer.output[b][i]*ReLu_Derivative(Current_Layer.value[b][j])
                elif Current_Layer.activation_function == 'Sigmoid':
                    Current_Layer.weights[i][j] = Current_Layer.weights[i][j] - (1/Batch_Size)*learning_rate*Next_Layer.propagation_units[j][b]*Last_Layer.output[b][i]*Sigmoid_Derivative(Current_Layer.value[b][j])
    print()

    for i in range(np.shape(Current_Layer.bias)[0]):
        for b in range(Batch_Size):
            print('\r  Optimizing biases... '+str(i+1)+'/'+str(np.shape(Current_Layer.bias)[0])+' '+str(b+1)+'/'+str(Batch_Size), end='')
            if Current_Layer.activation_function == 'ReLu':
                Current_Layer.bias[i] = Current_Layer.bias[i] - (1/Batch_Size)*learning_rate*Next_Layer.propagation_units[i][b]*ReLu_Derivative(Current_Layer.value[b][i])
            elif Current_Layer.activation_function == 'Sigmoid':
                Current_Layer.bias[i] = Current_Layer.bias[i] - (1/Batch_Size)*learning_rate*Next_Layer.propagation_units[i][b]*Sigmoid_Derivative(Current_Layer.value[b][i])
    print()

    return Current_Layer.propagation_units, Current_Layer.weights, Current_Layer.bias
