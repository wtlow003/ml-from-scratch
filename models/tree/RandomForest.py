from collections import Counter

import numpy as np

from .DecisionTree import DecisionTreeClassifier


def _bootstrap_sample(X, y):
    """Conduct bootstrap sampling."""
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)

    return X[idxs], y[idxs]


def _most_common_label(y):
    """Find the most common label given predictions."""
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]

    return most_common


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int,
        min_samples_split: int = 2,
        max_depth: int = 100,
    ):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        n_feats = int(round(np.sqrt(X.shape[1])))
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=n_feats,
            )
            # fit tree with bootstrap samples
            X_sample, y_sample = _bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.estimators.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.estimators])
        # majority vote
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [_most_common_label(tree_pred) for tree_pred in tree_preds]

        return np.array(y_pred)
