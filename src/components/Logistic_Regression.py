import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr, max_iters, tol):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.w = None

    def fit(self, X, y):
        # Initializing weights and gradient descent
        self.w = np.zeros(X.shape[1])
        self.w[0] = 1

        for i in range(self.max_iters):
            # Computing the gradient descent
            grad_dis = np.matmul(X.transpose(), sigmoid(np.matmul(self.w, X.transpose())) - y)

            # Updating the weights using gradient descent
            self.w = self.w - self.lr * grad_dis

            # Compute the norm of the gradient descent
            grad_norm = np.linalg.norm(grad_dis)
            if grad_norm < self.tol:
                break

    def predict(self, P, threshold=0.5):
        probabilities = sigmoid(np.matmul(P, self.w))
        predictions = (probabilities >= threshold).astype(int)
        return predictions


    def get_params(self, deep=True):
        # Return a dictionary of parameters
        return {
            'lr': self.lr,
            'max_iters': self.max_iters,
            'tol': self.tol
        }
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self 


