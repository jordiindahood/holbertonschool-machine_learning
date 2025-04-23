#!/usr/bin/env python3
""" script 4 """
import numpy as np


def convolve_channels(images, kernel, padding="same", stride=(1, 1)):
    """
    Perform a convolution on a batch of images with multiple channels.
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    sw, sh = stride[1], stride[0]

    ph, pw = 0, 0

    if padding == "same":
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    padded_image = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    new_h = int(((padded_image.shape[1] - kh) / sh) + 1)
    new_w = int(((padded_image.shape[2] - kw) / sw) + 1)

    output = np.zeros((m, new_h, new_w))

    for x in range(new_w):
        for y in range(new_h):
            output[:, y, x] = (
                kernel * padded_image[:,
                                      y * sh: y * sh + kh,
                                      x * sw: x * sw + kw,
                                      :]
            ).sum(axis=(1, 2, 3))

    return output
