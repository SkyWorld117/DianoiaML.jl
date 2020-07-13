import numpy as np
import gc, random, copy, math
from tools.vectorize import Vectorize
from tools.hex_bin_converter import Hex2Bin, Bin2Hex
from loss_function.standard_loss import Standard_Loss
from optimizer.gradient_descent import Gradient_Descent

def Forward_Propagation(sequential):
    for i in range(1, len(sequential.layers)):
        sequential.layers[i].run(sequential.layers[i-1].output)
    return sequential

connector = 0

def Weights_Mutation(Q, start_position, batch_size, weights):
    global connector
    multified_weights = np.zeros(np.shape(weights))
    for i in range(start_position, start_position+batch_size):
        for j in range(np.shape(weights)[1]):
            if random.choices([True, False], weights=(connector, 1-connector), k=2):
                if Hex2Bin(weights[i][j])=='Failed':
                    continue
                s = list(Hex2Bin(weights[i][j]))
                t = random.randint(0, len(s)-1)
                while s[t]=='.':
                    t = random.randint(0, len(s)-1)
                if s[t]=='0':
                    s[t] = '1'
                elif s[t]=='1':
                    s[t] = '0'
                z = ''
                for k in s:
                    z += k
                multified_weights[i][j] = Bin2Hex(z)
    Q.put(multified_weights)

def Biases_Mutation(biases):
    global connector
    for i in range(len(biases)):
        if random.choices([True, False], weights=(connector, 1-connector), k=2):
            if Hex2Bin(biases[i])=='Failed':
                continue
            s = list(Hex2Bin(biases[i]))
            t = random.randint(0, len(s)-1)
            while s[t]=='.':
                t = random.randint(0, len(s)-1)
            if s[t]=='0':
                s[t] = '1'
            elif s[t]=='1':
                s[t] = '0'
            z = ''
            for k in s:
                z += k
            biases[i] = Bin2Hex(z)
    return biases

class Hidden_Output_Layer:

    def __init__(self):
        self.propagation_units = []

def Genetic_Algorithm(sequential=None, input_data=[], output_data=[], loss_function=None, mutation_rate=0.01, adaptive_mutation_rate=True, epigenetics=True, population=50, epochs=6000, output=True):
    batch_size = np.shape(input_data)[0]
    sequential.layers[0].output = input_data
    HOL = Hidden_Output_Layer()

    global connector
    connector = mutation_rate

    min_error = math.inf

    if output:
        for e in range(epochs):
            print('Epoch', e+1)
            pools = []
            pools.append(copy.deepcopy(sequential))
            if epigenetics:
                pools[0] = Forward_Propagation(pools[0])
                pools[0] = Gradient_Descent(sequential=pools[0], input_data=input_data, output_data=output_data, loss_function=loss_function, learning_rate=0.02, epochs=1, output=False)
                pools[0] = Forward_Propagation(pools[0])
                pools[0].layers.append(HOL)
                pools[0].layers[len(pools[0].layers)-1].propagation_units = Vectorize(loss_function.propagation, matrix_a=pools[0].layers[len(pools[0].layers)-2].output, matrix_b=output_data)
                error = abs(np.sum(pools[0].layers[len(pools[0].layers)-1].propagation_units))
                del pools[0].layers[len(pools[0].layers)-1]
            else:
                error = min_error
            best = 0
            for i in range(1, population):
                print('\r Creating and evaluating network', i+1, end='')
                pools.append(copy.deepcopy(sequential))
                for l in range(1, len(pools[i].layers)):
                    pools[i].layers[l].weights = Vectorize(Weights_Mutation, matrix_a=pools[i].layers[l].weights)
                    pools[i].layers[l].biases = Biases_Mutation(pools[i].layers[l].biases)
                if epigenetics:
                    pools[i] = Forward_Propagation(pools[i])
                    pools[i] = Gradient_Descent(sequential=pools[i], input_data=input_data, output_data=output_data, loss_function=loss_function, learning_rate=0.02, epochs=1, output=False)
                pools[i] = Forward_Propagation(pools[i])
                pools[i].layers.append(HOL)
                pools[i].layers[len(pools[i].layers)-1].propagation_units = Vectorize(loss_function.propagation, matrix_a=pools[i].layers[len(pools[i].layers)-2].output, matrix_b=output_data)

                if error>abs(np.sum(pools[i].layers[len(pools[i].layers)-1].propagation_units)):
                    error = abs(np.sum(pools[i].layers[len(pools[i].layers)-1].propagation_units))
                    best = i

                del pools[i].layers[len(pools[i].layers)-1]

            sequential = pools[best]
            if adaptive_mutation_rate and (abs(error-min_error)<0.00001):
                connector *= 1.5
            min_error = error
            print('\n The chosen one is network', best)
            print(' Current error is', min_error)

            gc.collect()

    if not output:
        for e in range(epochs):
            pools = []
            pools.append(copy.deepcopy(sequential))
            if epigenetics:
                pools[0] = Forward_Propagation(pools[0])
                pools[0] = Gradient_Descent(sequential=pools[0], input_data=input_data, output_data=output_data, loss_function=loss_function, learning_rate=0.02, epochs=1, output=False)
                pools[0] = Forward_Propagation(pools[0])
                pools[0].layers.append(HOL)
                pools[0].layers[len(pools[0].layers)-1].propagation_units = Vectorize(loss_function.propagation, matrix_a=pools[0].layers[len(pools[0].layers)-2].output, matrix_b=output_data)
                error = abs(np.sum(pools[0].layers[len(pools[0].layers)-1].propagation_units))
                del pools[0].layers[len(pools[0].layers)-1]
            else:
                error = min_error
            best = 0
            for i in range(1, population):
                pools.append(copy.deepcopy(sequential))
                for l in range(1, len(pools[i].layers)):
                    pools[i].layers[l].weights = Vectorize(Weights_Mutation, matrix_a=pools[i].layers[l].weights)
                    pools[i].layers[l].biases = Biases_Mutation(pools[i].layers[l].biases)
                if epigenetics:
                    pools[i] = Forward_Propagation(pools[i])
                    pools[i] = Gradient_Descent(sequential=pools[i], input_data=input_data, output_data=output_data, loss_function=loss_function, learning_rate=0.02, epochs=1, output=False)
                pools[i] = Forward_Propagation(pools[i])
                pools[i].layers.append(HOL)
                pools[i].layers[len(pools[i].layers)-1].propagation_units = Vectorize(loss_function.propagation, matrix_a=pools[i].layers[len(pools[i].layers)-2].output, matrix_b=output_data)

                if error>abs(np.sum(pools[i].layers[len(pools[i].layers)-1].propagation_units)):
                    error = abs(np.sum(pools[i].layers[len(pools[i].layers)-1].propagation_units))
                    best = i

                del pools[i].layers[len(pools[i].layers)-1]

            sequential = pools[best]
            if adaptive_mutation_rate and (abs(error-min_error)<0.00001):
                connector *= 1.5
            min_error = error

            gc.collect()

    return sequential