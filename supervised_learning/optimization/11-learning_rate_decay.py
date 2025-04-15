#!/usr/bin/env python3
""" script 11 """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in NumPy.
    """

    decayed = alpha / (1 + decay_rate * int(global_step / decay_step))
    return decayed
