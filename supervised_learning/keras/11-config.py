#!/usr/bin/env python3
""" script 11 """

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.
    """
    json_string = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_string)
    return None


def load_config(filename):
    """
    Loads a model’s configuration from a JSON file.
    """
    with open(filename, "r") as f:
        network_string = f.read()
    network = K.models.model_from_json(network_string)
    return network
