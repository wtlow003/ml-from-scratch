import sys
from collections import Counter

sys.path.append("../")

import numpy as np
from utils import distances


class KNearestNeighbors:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray):
        pred_labels = [self._predict(x) for x in X]
        return np.array(pred_labels)

    def _predict(self, x):
        """Generate prediction for a single observation."""
        # compute distances
        computed_distances = [
            distances.euclidean_distance(x, x_train) for x_train in self.X_train
        ]
        # get k-nearest samples, labels
        k_indices = np.argsort(computed_distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority cls within the neighborhood
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]
