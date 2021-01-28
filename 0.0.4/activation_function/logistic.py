import numpy as np
from math import exp
from copy import deepcopy

def Logistic(input):
    return 1/(1+exp(-input))

def Logistic_Derivative(input):
    return (exp(-input))/((1+exp(-input))**2)
