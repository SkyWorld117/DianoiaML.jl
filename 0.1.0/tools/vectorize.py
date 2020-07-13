import multiprocessing
import numpy as np

def Vectorize(function, matrix_a=[], matrix_b=[]):
    Q = multiprocessing.Queue()
    processes = []
    threads = multiprocessing.cpu_count()
    batch_size = np.shape(matrix_a)[0]
    a = batch_size//threads
    b = batch_size%threads
    start_position = 0

    if np.shape(matrix_b)!=(0,):
        for i in range(threads):
            if b>0:
                processes.append(multiprocessing.Process(target=function, args=(Q, start_position, a+1, matrix_a, matrix_b)))
                b = b-1
                start_position = start_position+a+1
            else:
                processes.append(multiprocessing.Process(target=function, args=(Q, start_position, a, matrix_a, matrix_b)))
                start_position = start_position+a
            processes[i].start()

        result = Q.get()
        for i in range(threads-1):
            result = result+Q.get()
        for i in range(threads):
            processes[i].join()
        return result

    if np.shape(matrix_b)==(0,):
        for i in range(threads):
            if b>0:
                processes.append(multiprocessing.Process(target=function, args=(Q, start_position, a+1, matrix_a)))
                b = b-1
                start_position = start_position+a+1
            else:
                processes.append(multiprocessing.Process(target=function, args=(Q, start_position, a, matrix_a)))
                start_position = start_position+a
            processes[i].start()

        result = Q.get()
        for i in range(threads-1):
            result = result+Q.get()
        for i in range(threads):
            processes[i].join()
        return result
        print(result)
