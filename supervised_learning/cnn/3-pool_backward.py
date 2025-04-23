#!/usr/bin/env python3
""" script 3 """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs the backward pass for a pooling layer.
    """

    m, h_new, w_new, c_new = dA.shape

    m, _, _, _ = A_prev.shape()

    kh, kw = kernel_shape

    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)

    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for v in range(c_new):
                    pool = A_prev[z, y * sh:(kh+y*sh), x * sw:(kw+x*sw), v]
                    dA_aux = dA[z, y, x, v]
                    if mode == 'max':
                        z_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(z_mask, pool == _max, 1)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), v] += z_mask * dA_aux
                    if mode == 'avg':
                        avg = dA_aux / kh / kw
                        o_mask = np.ones(kernel_shape)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), v] += o_mask * avg
    return dA_prev
