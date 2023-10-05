import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Define the neural network class
class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units, alpha, iterations):
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.iterations = iterations

    def init_params(self,Shape):
        np.random.seed(0)
        self.W1 = np.random.rand(self.hidden_units, Shape) - 0.5
        self.b1 = np.random.rand(self.hidden_units, 1) - 0.5
        self.W2 = np.random.rand(self.hidden_units, self.hidden_units) - 0.5
        self.b2 = np.random.rand(self.hidden_units, 1) - 0.5

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z))  # Subtract max(Z) for numerical stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)

    def backward_prop(self, X, Y):
        m = X.shape[1]
        one_hot_Y = self.one_hot(Y)
        dZ2 = self.A2 - one_hot_Y
        self.dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        self.db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * self.ReLU_deriv(self.Z1)
        self.dW1 = (1 / m) * np.dot(dZ1, X.T)
        self.db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    def update_params(self):
        self.W1 -= self.alpha * self.dW1
        self.b1 -= self.alpha * self.db1
        self.W2 -= self.alpha * self.dW2
        self.b2 -= self.alpha * self.db2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        Y = Y.astype(int)
        one_hot_Y = np.zeros((Y.size, self.hidden_units))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def fit(self, X, Y):
        self.init_params(X.shape[0])
        for i in range(self.iterations):
            self.forward_prop(X)
            self.backward_prop(X, Y)
            self.update_params()

        return self

    def predict(self, X):
        self.forward_prop(X)
        return np.argmax(self.A2, axis=0)