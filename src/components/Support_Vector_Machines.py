import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class Support_Vector_Machine(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.svm_classifier = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, random_state=self.random_state)

    def fit(self, X_train, y_train):
        self.svm_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm_classifier.predict(X_test)
