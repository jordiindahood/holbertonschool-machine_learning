#!/usr/bin/env python3
""" Task 0: 0. Valid Convolution """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.
    """
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    filter = np.zeros((images.shape[0], int(h - kh + 1), int(w - kw + 1)))
    for i in range(int(w - kw + 1)):
        for j in range(int(h - kh + 1)):
            filter[:, j, i] = (kernel * images[:, j: j + kh, i: i + kw]).sum(
                axis=(1, 2)
            )
    return filter
