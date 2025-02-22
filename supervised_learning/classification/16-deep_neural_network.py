#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """
    deep neural network
    """

    def __init__(self, nx, layers):
        """
        init
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                x = np.sqrt(2 / nx)
                self.weights['W1'] = np.random.randn(layers[i], nx) * x
            else:
                x = np.sqrt(2 / layers[i - 1])
                self.weights[W_key] = np.random.randn(layers[i],
                        layers[i - 1]) * x
