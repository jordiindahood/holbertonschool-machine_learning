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

        for idx in range(self.L):
            if not isinstance(layers[idx], int) or layers[idx] < 1:
                raise TypeError("layers must be a list of positive integers")
            layer_size = layers[idx - 1]
            prev_size = nx if idx == 1 else layers[idx - 2]
            self.weights[f"W{idx}"] = np.random.randn(
                layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.weights[f"b{idx}"] = np.zeros((layer_size, 1))
