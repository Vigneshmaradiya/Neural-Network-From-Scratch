import numpy as np


def mean_squared_error(predicted, expected):
    N = predicted.shape[0]
    return 2 * (predicted - expected) / N

def log_loss(predicted, expected):
    predicted = np.clip(predicted, 1e-100, 1-(1e-100)) # prevent terms from becoming 0 or inf
    return ((1 - expected) / (1 - predicted)) - (expected / predicted)

def categorical_cross_entropy_loss(predicted, expected):
    predicted = np.clip(predicted, 1e-100, 1-(1e-100))
    return -expected / predicted