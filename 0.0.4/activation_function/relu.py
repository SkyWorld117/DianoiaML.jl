import numpy as np
from copy import deepcopy

def ReLu(input):
    if input <= 0:
        return 0
    if input > 0:
        return input

def ReLu_Derivative(input):
    if input <= 0:
        return 0
    if input > 0:
        return 1
