#from math import exp, tanh
import math

def sigmoid(x):
    """ The logistical sigmoid function """
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    """ The derivative of sigmoid function """
    return x * (1 - x)

def leaky_relu(x):
    """ A Leaky ReLU function """
    if x > 0:
        return x
    else:
        return 0.01 * x

def d_leaky_relu(x):
    """ The derivative of Leaky ReLU function """
    return (1.0 if x >= 0 else 0.01)

def relu(x):
    """ The ReLU function """
    return max(0, x)

def d_relu(x):
    """ The derivative of ReLU function """
    return (1.0 if x >= 0 else 0.0)

def tanh(x):
    """ The Tanh function.
    Not used. Only here for ilustration.
    the network calls the math function which is faster.
    """
    return math.tanh(x)


def d_tanh(x):
    """ The derivative of Tanh function """
    return 1 - (math.tanh(x) ** 2)
