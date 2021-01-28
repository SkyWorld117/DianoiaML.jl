import numpy as np
from copy import deepcopy

def ReLu(input):
    if input <= 0:
        return 0
    if input > 0:
        return input

def ReLu_Derivative(input):
    if input <= 0:
        return 0
    if input > 0:
        return 1

def Vectorized_ReLu(input_matrix):
    output_matrix = deepcopy(input_matrix)
    for i in range(np.shape(output_matrix)[0]):
        for j in range(np.shape(output_matrix)[1]):
            output_matrix[i][j] = ReLu(output_matrix[i][j])
    return output_matrix

def Vectorized_ReLu_Derivative(input_matrix):
    output_matrix = deepcopy(input_matrix)
    for i in range(np.shape(output_matrix)[0]):
        for j in range(np.shape(output_matrix)[1]):
            output_matrix[i][j] = ReLu_Derivative(output_matrix[i][j])
    return output_matrix
