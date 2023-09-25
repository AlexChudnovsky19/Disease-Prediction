import numpy as np
#import pandas as pd
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        
        if len(set(y)) == 1:
            return {'leaf': True, 'class': y[0]}
        if len(X) == 0:
            return {'leaf': True, 'class': max(set(y), key=y.count)}

        # Find the best split based on information gain
        best_split = self._find_best_split(X, y)
    
        if best_split is None:
            return {'leaf': True, 'class': max(set(y), key=y.count)}

        # Recursively build subtrees
        left_X, left_y, right_X, right_y, split_feature, split_value = best_split
        left_subtree = self._build_tree(left_X, left_y)
        right_subtree = self._build_tree(right_X, right_y)

        return {'leaf': False, 'split_feature': split_feature,
                'split_value': split_value, 'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None

        num_parent = [np.sum(y == c) for c in np.unique(y)]
        entropy_parent = self._entropy(num_parent)

        best_info_gain = -1
        best_split = None

        for col in range(n):
            feature_values = np.unique(X[:, col])
            for value in feature_values:
                left_mask = X[:, col] <= value
                right_mask = X[:, col] > value

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                num_left = [np.sum(y[left_mask] == c) for c in np.unique(y)]
                num_right = [np.sum(y[right_mask] == c) for c in np.unique(y)]

                entropy_left = self._entropy(num_left)

                entropy_right = self._entropy(num_right)

                p_left = np.sum(left_mask) / m
                p_right = np.sum(right_mask) / m
                info_gain = entropy_parent - (p_left * entropy_left + p_right * entropy_right)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = (X[left_mask], y[left_mask], X[right_mask], y[right_mask], col, value)

        return best_split

    def _entropy(self, num_class):
        num_class = np.array(num_class)
        p = num_class / np.sum(num_class)
        return -np.sum(p * np.log2(p + 1e-10))

    def predict(self, X):
        predictions = [self._predict(x, self.tree) for x in X]
        return predictions

    def _predict(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['split_feature']] <= node['split_value']:
            return self._predict(x, node['left'])
        else:
            return self._predict(x, node['right'])
        
    def set_params(self, **params):
        # No hyperparameters to set in this simple Naive Bayes classifier
        return self

    def get_params(self, deep=True):
        # No hyperparameters to get in this simple Naive Bayes classifier
        return {}