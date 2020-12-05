import numpy as np

def Function(inputs):
    return np.exp(inputs-max(inputs))/sum(np.exp(inputs-max(inputs)))

class Softmax:

    def function(Q, start_position, batch_size, value_matrix):
        output_matrix = np.zeros(np.shape(value_matrix))
        for i in range(start_position, start_position+batch_size):
            output_matrix[i] = Function(value_matrix[i])
        Q.put(output_matrix)

    def derivative(inputs, position):
        outputs = Function(inputs)
        derivative_vector = np.zeros(np.shape(inputs))
        for i in range(len(outputs)):
            if i!=position:
                derivative_vector[i] = -outputs[i]*outputs[position]
            else:
                derivative_vector[i] = outputs[i]*(1-outputs[i])
        return derivative_vector