import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains juke hence will be faster to allocate than zeros
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result

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


