import numpy as np
from copy import deepcopy

def Quadratic_Loss(output, sample):
    return (output-sample)**2

def Quadratic_Loss_Propagation_Units(output, sample):
    return 2*(output-sample)

def Vectorized_Quadratic_Loss(output_matrix, sample_matrix):
    loss_matrix = deepcopy(output_matrix)
    for i in range(np.shape(output_matrix)[0]):
        for j in range(np.shape(output_matrix)[1]):
            loss_matrix[i][j] = Quadratic_Loss(output_matrix[i][j], sample_matrix[i][j])
    return loss_matrix

def Vectorized_Quadratic_Loss_Propagation_Units(output_matrix, sample_matrix):
    propagation_units = deepcopy(output_matrix)
    for i in range(np.shape(output_matrix)[0]):
        for j in range(np.shape(output_matrix)[1]):
            propagation_units[i][j] = Quadratic_Loss_Propagation_Units(output_matrix[i][j], sample_matrix[i][j])
    return propagation_units.T
