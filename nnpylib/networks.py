""" 
Python implementation of a simple Feedforward Neural Network.
This version allows multiple hidden layers.


    Author: Gustavo Selbach Teixeira

"""
import random
import math
from nnpylib.nonlinear import (leaky_relu, d_leaky_relu,
                               relu, d_relu,
                               sigmoid, d_sigmoid,
                               tanh, d_tanh)


class Layer(object):
    """ 
    The layer object class.
    
    Attributes:
    -----------
    values : list
        This is where we write and read data from the NN.
    bias : list
        Used to compute the predicitons.
    deltas : list
        Used to correct the errors though the network.
    weights : list of lists - matrix
        Set the weights of the connections
    """
    def __init__(self, n_nodes:int, n_synapses:int=0):
        """ setup nn layers
        :param n_nodes: How many neurons this layer holds.
        :param n_synapses: The number of neurons from the previous layer.
        """
        self.n_nodes = n_nodes
        self.n_synapses = n_synapses
        self.values = [random.uniform(0, 1) for i in range(n_nodes)]
        self.bias = [random.uniform(0, 1) for i in range(n_nodes)]
        self.deltas = [random.uniform(0, 1) for i in range(n_nodes)]
        #initialize weights (synapses)
        self.weights = []
        for i in range(n_synapses):
            row = []
            for j in range(n_nodes):
                row.append(random.uniform(0, 1))
            self.weights.append(row)

class NeuralNetwork(object):
    """
    The Neural Network object. Holds the layers.
    
    Attributes:
    -----------
        input_layer : Layer
            Where data is inserted to the network.
        hidden_layers : list of Layers
            The list of Hidden Layers.
        output_layer : Layer
            Where data is read.
        learning_rate : float
            The rate of the learning process.
        nonlinear_method : str
            The non linear method to use as logistical function.
    """
    def __init__(self, inputs:int, outputs:int, hidden:list,
                 learning_rate:float=0.1, method:str="leaky_relu"):
        """ setup nn layers
        :param inputs: the number of neurons in the input layer
        :param outputs: the number of neurons in the output layer
        :param hidden: a list with the number of neurons for each 
                        hidden layer
        :param learning_rate: the speed of learning
        :param method: the logistical function to use
        """
        self.input_layer = Layer(inputs, 0)
        self.hidden_layers_number = len(hidden)
        last_size = inputs
        self.hidden_layers = []
        for hid in hidden:
            self.hidden_layers.append(Layer(hid, last_size))
            last_size = hid
        self.output_layer = Layer(outputs, last_size)
        self.learning_rate = learning_rate
        # Define the logistical method
        self.nonlinear_method = method
        if method == "leaky_relu":
            self._nonlinear_function = leaky_relu
            self._d_nonlinear_function = d_leaky_relu
        elif method == "relu":
            self._nonlinear_function = relu
            self._d_nonlinear_function = d_relu
        elif method == "sigmoid":
            self._nonlinear_function = sigmoid
            self._d_nonlinear_function = d_sigmoid
        elif method == "tanh":
            self._nonlinear_function = math.tanh
            self._d_nonlinear_function = d_tanh
        else:
            raise Exception("Invalid logistical method: " + method)

    def set_input(self, input_params:list):
        """ Feed the network with data. """
        for i in range(self.input_layer.n_nodes):
            self.input_layer.values[i] = input_params[i]

    def activation_function(self, source, target):
        """ The Activation function """
        for j in range(target.n_nodes):
            activation = target.bias[j]
            for k in range(source.n_nodes):
                activation += (source.values[k] * target.weights[k][j])
            target.values[j] = self._nonlinear_function(activation)

    def calc_loss(self, expected:list):
        """ Compute the errors for the output layer """
        for i in range(len(self.output_layer.values)):
            error = (expected[i] - self.output_layer.values[i])
            self.output_layer.deltas[i] = (error 
                    * self._d_nonlinear_function(self.output_layer.values[i]))

    def calc_deltas(self, source, target):
        """ Compute the deltas between layers """
        for j in range(target.n_nodes):
            error = 0.0
            for k in range(source.n_nodes):
                error += (source.deltas[k] * source.weights[j][k])
            target.deltas[j] = (error 
                        * self._d_nonlinear_function(target.values[j]))

    def update_weights(self, source, target):
        """ Update the weights """
        for j in range(source.n_nodes):
            source.bias[j] += (source.deltas[j] * self.learning_rate)
            for k in range(target.n_nodes):
                source.weights[k][j] += (target.values[k]
                                        * source.deltas[j] * self.learning_rate)

    def forward_pass(self):
        """ NN Activation step """
        k = 0
        self.activation_function(self.input_layer, self.hidden_layers[k])
        # Run through the hidden layers. If theres more than 1.
        last_hidden_layer = self.hidden_layers_number - 1
        while k < last_hidden_layer:
            self.activation_function(self.hidden_layers[k], 
                                self.hidden_layers[k+1])
            k += 1
        self.activation_function(self.hidden_layers[k], self.output_layer)
        
    def back_propagation(self, outputs:list):
        """ The back propagation process.
        Computes the deltas and update the weights and bias.
        If there' multiple hidden_layers, loops back though then
        """
        last_hidden_layer = self.hidden_layers_number - 1
        k = last_hidden_layer
        self.calc_deltas(self.output_layer, self.hidden_layers[k])
        self.update_weights(self.output_layer, self.hidden_layers[k])
        while k > 0:
            self.calc_deltas(self.hidden_layers[k], self.hidden_layers[k-1])
            self.update_weights(self.hidden_layers[k],
                                self.hidden_layers[k-1])
            k -= 1
        self.update_weights(self.hidden_layers[k], self.input_layer)
        
    def train(self, inputs:list, outputs:list, n_iteractions:int):
        """ Training main loop """
        num_training_sets = len(outputs)
        training_sequence = list(range(num_training_sets))
        for n in range(n_iteractions):
            random.shuffle(training_sequence)
            for x in range(num_training_sets):
                i = training_sequence[x]
                self.set_input(inputs[i])
                # forward activation
                self.forward_pass()
                print("{} Input: {} Expected: {} Output: {}".format(n, 
                                                    inputs[i],
                                                    outputs[i],
                                                    self.output_layer.values))
                self.calc_loss(outputs[i])
                self.back_propagation(outputs[i])

    def predict(self, inputs:list):
        """ Make a prediction. To be used after the network is trained """
        self.set_input(inputs)
        self.forward_pass()
        return self.output_layer.values


    


    
