import numpy as np


def euclidean_distance(x1, x2):
    """Compute euclidean distances between two samples."""
    return np.sqrt(np.sum(x1 - x2) ** 2)
