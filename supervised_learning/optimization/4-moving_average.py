#!/usr/bin/env python3
""" script 4 """


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.
    """
    myData = []
    x = 0
    for i in range(len(data)):
        x = beta * x + (1 - beta) * data[i]
        myData.append(x / (1 - beta ** (i + 1)))

    return myData
