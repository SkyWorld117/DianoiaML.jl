import numpy as np
from math import exp

def Function(input):
    if input>=0:
        return 1/(1+exp(-input))
    else:
        return exp(input)/(1+exp(input))

def Derivative(input):
    return (exp(-input))/((1+exp(-input))**2)

class Logistic:

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

# Source: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/