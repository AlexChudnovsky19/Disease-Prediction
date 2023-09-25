import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist

class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_arg in X_test:
            # Computing the Euclidean distance for each test argument
            distances = cdist([test_arg], self.X_train, metric='euclidean')[0]

            # Storing the index's of the k smallest distances
            k_smallest = np.argsort(distances)[:self.k]

            # Saving the k nearest labels using the k smallest distances index's
            k_nearest_labels = self.y_train[k_smallest]

            # Computing the labels and their counts
            label_count = {}
            for label in k_nearest_labels:
                if label in label_count:
                   label_count[label] += 1
                else:
                   label_count[label] = 1

            # Choosing the label with the nearest samples
            max_label = max(label_count, key=label_count.get)
            predictions.append(max_label)

        return predictions
