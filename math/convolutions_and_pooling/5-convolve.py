#!/usr/bin/env python3
""" script 5 """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform a convolution operation on a batch of images using
    multiple kernels.
    """
    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    nc, kw, kh = kernels.shape[3], kernels.shape[1], kernels.shape[0]
    sw, sh = stride[1], stride[0]

    pw, ph = 0, 0

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        # Extract required padding
        ph = padding[0]
        pw = padding[1]

    # pad images
    images = np.pad(images,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    new_h = int(((h + 2 * ph - kh) / sh) + 1)
    new_w = int(((w + 2 * pw - kw) / sw) + 1)

    # initialize convolution output tensor
    output = np.zeros((m, new_h, new_w, nc))

    # Loop over every pixel of the output
    for y in range(new_h):
        for x in range(new_w):
            # over every kernel
            for v in range(nc):
                # element-wise multiplication of the kernel and the image
                output[:, y, x, v] = \
                    (kernels[:, :, :, v] *
                     images[:,
                     y * sh: y * sh + kh,
                     x * sw: x * sw + kw,
                     :]).sum(axis=(1, 2, 3))

    return output
