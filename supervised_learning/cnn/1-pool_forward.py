#!/usr/bin/env python3
""" script 1"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation for a pooling layer.
    """

    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    pw, ph = 0, 0

    n_h = int(((h_prev - kh) / sh) + 1)
    n_w = int(((w_prev - kw) / sw) + 1)

    output = np.zeros((m, n_h, n_w, c_prev))

    for x in range(n_w):
        for y in range(n_h):
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(A_prev[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(A_prev[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw], axis=(1, 2))

    return output
