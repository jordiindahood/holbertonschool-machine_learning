#!/usr/bin/env python3
""" script 2 """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images with specified padding.
    """

    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    padded_image = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw)),
                          mode='constant', constant_values=0)
    H = int(padded_image.shape[1] - kh + 1)
    W = int(padded_image.shape[2] - kw + 1)
    image_filter = np.zeros((images.shape[0], H, W))
    for i in range(W):
        for j in range(H):
            image_filter[:, j, i] = (kernel *
                                     padded_image[:, j: j + kh,
                                                  i: i + kw]).\
                                                  sum(axis=(1, 2))
    return image_filter
