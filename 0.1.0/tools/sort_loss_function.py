import numpy as np

def Loss_Function(Q, start_position, batch_size, output_matrix, sample_matrix):
    loss = 0
    for i in range(start_position, start_position+batch_size):
        if np.argmax(output_matrix[i]) != np.argmax(sample_matrix[i]):
            loss += 1
    Q.put(loss)
