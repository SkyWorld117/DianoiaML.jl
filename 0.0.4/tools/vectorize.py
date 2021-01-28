import multiprocessing
import numpy as np

def Loss_Function(Q, function, start_position, batch_size, output_matrix, sample_matrix):
    loss_matrix = np.zeros(np.shape(output_matrix))
    for i in range(start_position, start_position+batch_size):
        for j in range(np.shape(loss_matrix)[1]):
            loss_matrix[i][j] = function(output_matrix[i][j], sample_matrix[i][j])
    Q.put(loss_matrix)

def Loss_Function_Propagation_Units(Q, function, start_position, batch_size, output_matrix, sample_matrix):
    propagation_units = np.zeros(np.shape(output_matrix))
    for i in range(start_position, start_position+batch_size):
        for j in range(np.shape(propagation_units)[1]):
            propagation_units[i][j] = function(output_matrix[i][j], sample_matrix[i][j])
    Q.put(propagation_units.T)

def Activation_Function(Q, function, start_position, batch_size, value_matrix):
    output_matrix = np.zeros(np.shape(value_matrix))
    for i in range(start_position, start_position+batch_size):
        for j in range(np.shape(output_matrix)[1]):
            output_matrix[i][j] = function(value_matrix[i][j])
    Q.put(output_matrix)

def Activation_Function_Derivative(Q, function, start_position, batch_size, value_matrix):
    derivative_matrix = np.zeros(np.shape(value_matrix))
    for i in range(start_position, start_position+batch_size):
        for j in range(np.shape(derivative_matrix)[1]):
            derivative_matrix[i][j] = function(value_matrix[i][j])
    Q.put(derivative_matrix)

def Vectorize(function, type, matrix_a=None, matrix_b=None):
    Q = multiprocessing.Queue()
    processes = []
    threads = multiprocessing.cpu_count()
    batch_size = np.shape(matrix_a)[0]
    a = batch_size//threads
    b = batch_size%threads
    start_position = 0

    if type == 'Loss_Function':
        for i in range(threads):
            if b>0:
                processes.append(multiprocessing.Process(target=Loss_Function, args=(Q, function, start_position, a+1, matrix_a, matrix_b)))
                b = b-1
                start_position = start_position+a+1
            else:
                processes.append(multiprocessing.Process(target=Loss_Function, args=(Q, function, start_position, a, matrix_a, matrix_b)))
                start_position = start_position+a
            processes[i].start()

        loss_matrix = np.zeros(np.shape(matrix_a))
        for i in range(threads):
            loss_matrix = loss_matrix+Q.get()
        for i in range(threads):
            processes[i].join()
        return loss_matrix

    if type == 'Loss_Function_Propagation_Units':
        for i in range(threads):
            if b>0:
                processes.append(multiprocessing.Process(target=Loss_Function_Propagation_Units, args=(Q, function, start_position, a+1, matrix_a, matrix_b)))
                b = b-1
                start_position = start_position+a+1
            else:
                processes.append(multiprocessing.Process(target=Loss_Function_Propagation_Units, args=(Q, function, start_position, a, matrix_a, matrix_b)))
                start_position = start_position+a
            processes[i].start()

        propagation_units = np.zeros(np.shape(matrix_a)).T
        for i in range(threads):
            propagation_units = propagation_units+Q.get()
        for i in range(threads):
            processes[i].join()
        return propagation_units

    if type == 'Activation_Function':
        for i in range(threads):
            if b>0:
                processes.append(multiprocessing.Process(target=Activation_Function, args=(Q, function, start_position, a+1, matrix_a)))
                b = b-1
                start_position = start_position+a+1
            else:
                processes.append(multiprocessing.Process(target=Activation_Function, args=(Q, function, start_position, a, matrix_a)))
                start_position = start_position+a
            processes[i].start()

        output_matrix = np.zeros(np.shape(matrix_a))
        for i in range(threads):
            output_matrix = output_matrix+Q.get()
        for i in range(threads):
            processes[i].join()
        return output_matrix

    if type == 'Activation_Function_Derivative':
        for i in range(threads):
            if b>0:
                processes.append(multiprocessing.Process(target=Activation_Function_Derivative, args=(Q, function, start_position, a+1, matrix_a)))
                b = b-1
                start_position = start_position+a+1
            else:
                processes.append(multiprocessing.Process(target=Activation_Function_Derivative, args=(Q, function, start_position, a, matrix_a)))
                start_position = start_position+a
            processes[i].start()

        derivative_matrix = np.zeros(np.shape(matrix_a))
        for i in range(threads):
            derivative_matrix = derivative_matrix+Q.get()
        for i in range(threads):
            processes[i].join()
        return derivative_matrix
