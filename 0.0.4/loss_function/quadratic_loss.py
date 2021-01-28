import numpy as np
from copy import deepcopy

def Quadratic_Loss(output, sample):
    return (output-sample)**2

def Quadratic_Loss_Propagation_Units(output, sample):
    return 2*(output-sample)
