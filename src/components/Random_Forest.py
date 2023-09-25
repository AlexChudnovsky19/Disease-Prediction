from Decision_Trees import DecisionTree
import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees = 20, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Randomly sample data with replacement (bootstrapping)
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]

            # Create a decision tree and fit it to the sampled data
            tree = DecisionTree()
            tree.fit(X_sampled, y_sampled)
            
            # Append the trained tree to the list of trees
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions using each tree in the forest and return the majority vote
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Calculate the majority vote for each sample
        majority_vote, _ = mode(all_predictions, axis=0)

        return majority_vote.ravel()  # Return the majority vote predictions as a 1D array
