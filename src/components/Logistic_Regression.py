import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'\Users\alexc\Heart_failure_predictor\notebook\heart_failure.csv')
X = df.drop('HeartDisease', axis=1)  # Separate the features (X) from the target variable
y = df['HeartDisease']  # Set 'y' to be the target variable column

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),        
    ]
)

X = df.drop('HeartDisease', axis=1)  # Separate the features (X) from the target variable
y = df['HeartDisease']  # Set 'y' to be the target variable column

X = preprocessor.fit_transform(X)  # Fit and transform the features (X)
y.view()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr, max_iters=30000, tol=0.005):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.w = None

    def fit(self, P, y):
        # Initializing weights and gradient descent
        self.w = np.zeros(P.shape[1])
        self.w[0] = 1

        for i in range(self.max_iters):
            # Computing the gradient descent
            grad_dis = np.matmul(P.transpose(), sigmoid(np.matmul(self.w, P.transpose())) - y)

            # Updating the weights using gradient descent
            self.w = self.w - self.lr * grad_dis

            # Compute the norm of the gradient descent
            grad_norm = np.linalg.norm(grad_dis)
            if grad_norm < self.tol:
                break

    def predict(self, P):
        return sigmoid(np.matmul(P, self.w))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Create custom functions for k-fold cross-validation
def split_data_into_folds(X, y, k):
    fold_size = len(X) // k
    folds_X = []
    folds_y = []

    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size

        X_fold = X[start_idx:end_idx]
        y_fold = y[start_idx:end_idx]

        folds_X.append(X_fold)
        folds_y.append(y_fold)

    return folds_X, folds_y

def k_fold_cross_validation(X, y, k, model):
    accuracies = []

    folds_X, folds_y = split_data_into_folds(X, y, k)

    for i in range(k):
        X_train = np.concatenate(folds_X[:i] + folds_X[i+1:], axis=0)
        y_train = np.concatenate(folds_y[:i] + folds_y[i+1:], axis=0)
        X_test, y_test = folds_X[i], folds_y[i]

        # Initialize the model
        model_instance = model

        # Fit the model on the training data
        model_instance.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = model_instance.predict(X_test)

        # Convert predicted probabilities to binary labels using a threshold of 0.5
        predicted_labels = (predictions >= 0.5).astype(int)

        # Calculate accuracy for the current fold and store it
        accuracy = np.mean(predicted_labels == y_test)
        accuracies.append(accuracy)

    return accuracies

# Step 5: Perform k-fold cross-validation and calculate mean accuracy
lr = 0.05
model = LogisticRegression(lr)
k_folds = 10

# Perform k-fold cross-validation
accuracies = k_fold_cross_validation(X_train, y_train, k_folds, model)

# Calculate the mean and standard deviation of accuracy across all folds
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.2f}")

model.fit(X_train, y_train)

# Compute predictions for all test samples at once
predictions = model.predict(X_test)

# Calculate accuracy
predicted_labels = (predictions >= 0.5).astype(int)
accuracy = np.mean(predicted_labels == y_test)

y_true = np.array(y_test)
y_pred = np.array(predicted_labels)

    # Calculate squared differences
squared_diff = (y_true - y_pred) ** 2

    # Calculate mean of squared differences
mean_squared_diff = np.mean(squared_diff)

    # Take the square root to get RMSE
rmse_score = np.sqrt(mean_squared_diff)

    # Calculate absolute differences
absolute_diff = np.abs(y_true - y_pred)

    # Calculate mean of absolute differences
mae_score = np.mean(absolute_diff)

    # Calculate the total sum of squares (TSS)
tss = np.sum((y_true - np.mean(y_true))**2)

    # Calculate the residual sum of squares (RSS)
rss = np.sum((y_true - y_pred)**2)

    # Calculate R2 score
r2 = 1 - (rss / tss)

print(f"Accuracy: {accuracy:.2f}")
print(f"RMSE on the test set: {rmse_score:.2f}")
print(f"MSE on the test set: {mean_squared_diff:.2f}")
print(f"MAE on the test set: {mae_score:.2f}")
print(f"R2 on the test set: {r2:.2f}")

