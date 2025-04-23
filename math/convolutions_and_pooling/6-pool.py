#!/usr/bin/env python3
""" script 6 """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform pooling operation on a batch of images.

    """
    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    kw, kh = kernel_shape[1], kernel_shape[0]
    sw, sh = stride[1], stride[0]

    new_h = int(((h - kh) / sh) + 1)
    new_w = int(((w - kw) / sw) + 1)

    # initialize convolution output tensor
    output = np.zeros((m, new_h, new_w, c))

    # Loop over every pixel of the output
    for x in range(new_w):
        for y in range(new_h):
            # element-wise multiplication of the kernel and the image
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(images[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(images[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw], axis=(1, 2))

    return output
