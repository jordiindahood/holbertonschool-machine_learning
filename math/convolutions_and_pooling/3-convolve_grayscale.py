#!/usr/bin/env python3
""" script 3 """
import numpy as np


def convolve_grayscale(images, kernel, padding="same", stride=(1, 1)):
    """
    Perform a convolution on grayscale images.

    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    sw, sh = stride[1], stride[0]

    # calculates padding
    ph, pw = 0, 0

    if padding == "same":
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        # Extract required padding
        ph = padding[0]
        pw = padding[1]

    # image padding
    padded_image = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode="constant",
        constant_values=0,
    )
    new_h = int(((padded_image.shape[1] - kh) / sh) + 1)
    new_w = int(((padded_image.shape[2] - kw) / sw) + 1)

    # this is will form the shape of the output image
    output = np.zeros((m, new_h, new_w))

    # Loop over every pixel of the output
    for x in range(new_w):
        for y in range(new_h):
            # element-wise multiplication of the kernel and the image
            output[:, y, x] = (
                kernel * padded_image[:,
                                      y * sh: y * sh + kh,
                                      x * sw: x * sw + kw]
            ).sum(axis=(1, 2))

    return output
