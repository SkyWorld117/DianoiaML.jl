import numpy as np

def Function(output, sample):
    return (output-sample)**2

def Propagation_Unit(output, sample):
    return 2*(output-sample)

class Quadratic_Loss:

    def function(Q, start_position, batch_size, output_matrix, sample_matrix):
        loss_matrix = np.zeros(np.shape(output_matrix))
        for i in range(start_position, start_position+batch_size):
            for j in range(np.shape(loss_matrix)[1]):
                loss_matrix[i][j] = Function(output_matrix[i][j], sample_matrix[i][j])
        Q.put(np.sum(loss_matrix))

    def propagation(Q, start_position, batch_size, output_matrix, sample_matrix):
        propagation_units = np.zeros(np.shape(output_matrix))
        for i in range(start_position, start_position+batch_size):
            for j in range(np.shape(propagation_units)[1]):
                propagation_units[i][j] = Propagation_Unit(output_matrix[i][j], sample_matrix[i][j])
        Q.put(propagation_units.T)
