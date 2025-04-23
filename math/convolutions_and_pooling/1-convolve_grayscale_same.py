#!/usr/bin/env python3
""" script 1 """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images.
    """
    m, h, w = images.shape  # Number of images, height, and width
    kh, kw = kernel.shape   # Kernel height and width

    # padding for height and width
    pad_h = kh // 2
    pad_w = kw // 2

    # Apply zero padding to the images, accounting for even-sized kernels
    images_padded = np.pad(images,
                           pad_width=((0, 0), (pad_h, kh - pad_h - 1),
                                      (pad_w, kw - pad_w - 1)),
                           mode='constant', constant_values=0)

    # Initialize output array for the filtered images
    image_filter = np.zeros((m, h, w))

    # Perform the convolution operation
    for i in range(h):
        for j in range(w):
            image_filter[:, i, j] = (kernel *
                                     images_padded[:, i:i+kh, j:j+kw])\
                                         .sum(axis=(1, 2))

    return image_filter
