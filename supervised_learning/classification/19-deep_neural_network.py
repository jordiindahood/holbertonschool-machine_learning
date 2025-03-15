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

        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.__weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                x = np.sqrt(2 / nx)
                self.__weights['W1'] = np.random.randn(layers[i], nx) * x
            else:
                x = np.sqrt(2 / layers[i - 1])
                self.__weights[W_key] = np.random.randn(layers[i],
                                                        layers[i - 1]) * x

    @property
    def L(self):
        """ L"""
        return self.__L

    @property
    def cache(self):
        """ cache"""
        return self.__cache

    @property
    def weights(self):
        """ weight"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            A_key_prev = "A{}".format(i)
            A_key_forw = "A{}".format(i + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache
    
    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost