import numpy as np
from math import exp

def Function(input):
    return (exp(input)-exp(-input))/(exp(input)+exp(-input))

def Derivative(input):
    return 1-Function(input)**2

class tanh:

    def function(Q, start_position, batch_size, value_matrix):
        output_matrix = np.zeros(np.shape(value_matrix))
        for i in range(start_position, start_position+batch_size):
            for j in range(np.shape(output_matrix)[1]):
                output_matrix[i][j] = Function(value_matrix[i][j])
        Q.put(output_matrix)

    def derivative(inputs, position):
        derivative_vector = np.zeros(np.shape(inputs))
        derivative_vector[position] = Derivative(inputs[position])
        return derivative_vector