from collections import Counter

import numpy as np


class DecisionTreeClassifier:
    """Implementation of a Decision Tree Classifier."""

    class Node:
        """Implementation of Node in a Decision Tree."""

        def __init__(
            self, feature=None, threshold=None, left=None, right=None, *, value=None
        ):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def _is_leaf(self):
            """Check if current node is a leaf node."""
            return self.value is not None

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, criterion="gini"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.criterion = criterion
        self.root = None

    def _entropy(self, y):
        """Compute entropy."""
        hist = np.bincount(y)
        ps = hist / len(y)

        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        """Compute gini index."""
        hist = np.bincount(y)
        ps = hist / len(y)

        return 1.0 - np.sum([p ** 2 for p in ps if p > 0])

    def _most_common_value(self, y):
        """Retrieve the majority target class."""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]

        return most_common

    def _split(self, X_column, split_threshold):
        # return 1-dim array
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()

        return left_idxs, right_idxs

    def _information_gain(self, y, X_column, split_threshold):
        """Compute information gain."""
        # parent entropy
        parent_entropy = self._entropy(y)
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weight avg child entropy
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # return information gain
        return parent_entropy - child_entropy

    def _weighted_gini(self, y, X_column, split_threshold):
        """Compute weighted gini index."""
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)

        # weighted avg gini impurity
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        g_left, g_right = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        weighted_gini = (n_left / n) * g_left + (n_right / n) * g_right

        return weighted_gini

    def _best_split(self, X, y, feat_idxs):
        criterion = self.criterion
        best_gain = -1
        best_gini = np.inf
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                if criterion == "gini":
                    gini = self._weighted_gini(y, X_column, threshold)
                    if gini < best_gini:
                        best_gini = gini
                        split_idx = feat_idx
                        split_threshold = threshold

                if criterion == "entropy":
                    gain = self._information_gain(y, X_column, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold

        return split_idx, split_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        # retrieve the number of unique labels
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_value(y)
            return type(self).Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_feats, replace=False)

        # greedy search
        best_feat, best_threshold = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_threshold)
        # recursive binary split
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return type(self).Node(best_feat, best_threshold, left, right)

    def _traverse_tree(self, x, node):
        if node._is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        # grow the tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        # traverse the tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
