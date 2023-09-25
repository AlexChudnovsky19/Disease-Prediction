import numpy as np

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