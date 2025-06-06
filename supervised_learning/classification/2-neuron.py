#!/usr/bin/env python3
""" script 0"""
import numpy as np


class Neuron:
    """
    Neuron: Class
    """

    def __init__(self, nx):
        """
        Initialize
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Return Weight
        """
        return self.__W

    @property
    def b(self):
        """
        return bias
        """
        return self.__b

    @property
    def A(self):
        """
        return Activated output
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using sigmoid
        """
        Q = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Q))
        return self.__A
