#!/usr/bin/env python3
""" Task 1: 1. Normalize """
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    """
    result = (X - m) / s
    return result
