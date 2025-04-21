#!/usr/bin/env python3
""" Task 7: 7. Early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether to stop training early based on the validation cost.
    """
    count = 0 if opt_cost - cost - threshold > 0 else count + 1
    stop = False
    if (count == patience):
        stop = True

    return (stop, count)
