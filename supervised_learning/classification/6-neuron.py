#!/usr/bin/env python3
""" script 6"""
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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = cost / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        """
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neuron
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
