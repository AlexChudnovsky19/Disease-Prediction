import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # Initialize an empty dictionary to store feature summaries
        self.summaries = {}
    
    # Method to separate the dataset into classes based on the last column (class labels)
    def separate_by_class(self, dataset):
        separated = {}
        # Iterate through each instance in the dataset
        for i in range(len(dataset)):
            vector = dataset[i]
            if vector[-1] not in separated:
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    def mean(self, numbers):
        return np.mean(numbers)

    def stdev(self, numbers):
        return np.std(numbers)
    
    # Method to summarize a dataset by computing the mean and standard deviation for each feature
    def summarize_dataset(self, dataset):
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]  
        return summaries

    def summarize_by_class(self, dataset):
        separated = self.separate_by_class(dataset)
        summaries = {}
        for class_value, instances in separated.items():
            summaries[class_value] = self.summarize_dataset(instances)
        return summaries

    def calculate_probability(self,x, mean, stdev):
        if stdev == 0:
            return 0  # Handle division by zero gracefully, return 0 probability

        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        if math.isinf(exponent) or math.isnan(exponent):
            return 0  # Handle invalid values, return 0 probability

        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def fit(self, X, y):
        dataset = np.column_stack((X, y))
        self.summaries = self.summarize_by_class(dataset)

    # Method to predict the class labels of a set of input vectors (X)
    def predict(self, X):
        predictions = []
        for input_vector in X:
            probabilities = {}
            for class_value, class_summaries in self.summaries.items():
                probabilities[class_value] = 1
                for i in range(len(class_summaries)):
                    mean, stdev = class_summaries[i]
                    x = input_vector[i]
                    probabilities[class_value] *= self.calculate_probability(x, mean, stdev)
            prediction = max(probabilities, key=probabilities.get)
            predictions.append(prediction)
        return predictions

    def set_params(self, **params):
        # No hyperparameters to set in this simple Naive Bayes classifier
        return self

    def get_params(self, deep=True):
        # No hyperparameters to get in this simple Naive Bayes classifier
        return {}
