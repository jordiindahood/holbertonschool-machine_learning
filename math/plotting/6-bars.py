#!/usr/bin/env python3
"""
    bars.py
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    plot a stacked bar graph
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ["Farrah", "Fred", "Felicia"]
    arr1 = fruit[0]
    arr2 = fruit[1]
    arr3 = fruit[2]
    arr4 = fruit[3]
    plt.bar(names, arr1, width=0.5, color="red")
    plt.bar(names, arr2, width=0.5, bottom=arr1, color="yellow")
    plt.bar(names, arr3, width=0.5, bottom=arr1 + arr2, color="#ff8000")
    plt.bar(names, arr4, width=0.5, bottom=arr1 + arr2 + arr3, color="#ffe5b4")

    plt.ylim(0, 80)
    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.legend(["apples", "bananas", "oranges", "peaches"])
    plt.show()
