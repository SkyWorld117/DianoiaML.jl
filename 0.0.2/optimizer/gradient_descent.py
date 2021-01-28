# The sequence of backpropagation is Next_Layer to Current_Layer to Last_Layer.

import numpy as np
from activation_function.relu import Vectorized_ReLu_Derivative
from activation_function.logistic import Vectorized_Logistic_Derivative

def Gradient_Descent(learning_rate, batch_size, Current_Layer, Last_Layer, Next_Layer):
    if Current_Layer.activation_function == 'ReLu':
        derivative = Vectorized_ReLu_Derivative(Current_Layer.value)
    elif Current_Layer.activation_function == 'Logistic':
        derivative = Vectorized_Logistic_Derivative(Current_Layer.value)

    for j in range(Current_Layer.layer_size):
        for b in range(batch_size):
            for i in range(Current_Layer.input_size):
                print('\r  Opimizing propagation units, weights and biases... '+str(j+1)+'/'+str(Current_Layer.layer_size)+' '+str(b+1)+'/'+str(batch_size)+' '+str(i+1)+'/'+str(Current_Layer.input_size), end='')
                Current_Layer.propagation_units[i][j][b] = Current_Layer.weights[i][j]*derivative[b][j]*Next_Layer.propagation_units[j][b]
                Current_Layer.weights[i][j] = Current_Layer.weights[i][j] - (1/batch_size)*learning_rate*Next_Layer.propagation_units[j][b]*Last_Layer.output[b][i]*derivative[b][j]
            Current_Layer.bias[j] = Current_Layer.bias[j] - (1/batch_size)*learning_rate*Next_Layer.propagation_units[j][b]*derivative[b][j]
    Current_Layer.propagation_units = np.sum(Current_Layer.propagation_units, axis = 1)
    print()

    return Current_Layer.propagation_units, Current_Layer.weights, Current_Layer.bias
