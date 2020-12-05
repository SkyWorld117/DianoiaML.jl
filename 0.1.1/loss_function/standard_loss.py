import numpy as np

def Function(output, sample):
    return abs(output-sample)

class Standard_Loss:

    def function(Q, start_position, batch_size, output_matrix, sample_matrix):
        loss_matrix = np.zeros(np.shape(output_matrix))
        for i in range(start_position, start_position+batch_size):
            for j in range(np.shape(loss_matrix)[1]):
                loss_matrix[i][j] = Function(output_matrix[i][j], sample_matrix[i][j])
        Q.put(np.sum(loss_matrix))
