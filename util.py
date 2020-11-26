import numpy as np


def get_sample_weight(y):
    weight_per_class = {
        0: np.mean(y) / (1 - np.mean(y)),
        1: 1
    }
    return [weight_per_class[i] for i in y]
