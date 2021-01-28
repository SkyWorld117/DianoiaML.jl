import numpy as np
from math import exp
from copy import deepcopy

def Sigmoid(input):
    return 1/(1+exp(-input))

def Sigmoid_Derivative(input):
    return (exp(-input))/((1+exp(-input))**2)

def Vectorized_Sigmoid(input_matrix):
    output_matrix = deepcopy(input_matrix)
    for i in range(np.shape(output_matrix)[0]):
        for j in range(np.shape(output_matrix)[1]):
            output_matrix[i][j] = Sigmoid(output_matrix[i][j])
    return output_matrix

def Vectorized_Sigmoid_Derivative(input_matrix):
    output_matrix = deepcopy(input_matrix)
    for i in range(np.shape(output_matrix)[0]):
        for j in range(np.shape(output_matrix)[1]):
            output_matrix[i][j] = Sigmoid_Derivative(output_matrix[i][j])
    return output_matrix
