import os
import sys
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from Logistic_Regression import LogisticRegression
from Naive_Bayes import NaiveBayesClassifier
from Decision_Trees import DecisionTree
from K_Nearest_Neighbors import KNearestNeighbors
from Support_Vector_Machines import Support_Vector_Machine
from Random_Forest import RandomForest
from Neural_Network import NeuralNetwork
from src.utils import save_object,evaluate_models
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_directory = "artifacts"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, disease_name, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Logistic_Regression": LogisticRegression(lr=0.05, max_iters=20000, tol=0.001),
                "Naive_Bayes": NaiveBayesClassifier(),
                "Decision_Trees": DecisionTree(),
                "K_Nearest_Neighbors": KNearestNeighbors(k=20),
                "Support_Vector_Machines": Support_Vector_Machine(kernel='rbf', C=1.0, gamma='scale', random_state=None),
                "Random_Forest": RandomForest(),
                # "Neural Network": NeuralNetwork(hidden_units=10, alpha=0.1, iterations=1000)
            }
            params = {
                "Logistic_Regression": {},
                "Naive_Bayes": {},
                "Decision_Trees": {},
                "K_Nearest_Neighbors": {},
                "Support_Vector_Machines": {},
                "Random_Forest": {},
                # "Neural Network": {}
            }

            model_report: dict = evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                                models=models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"best model name is : {best_model_name}")

            # Adjust the file path based on the disease_name
            model_filename = os.path.join(self.model_trainer_config.trained_model_directory, f"model_{disease_name}.pkl")
             
            save_object(
                file_path=model_filename,
                obj=best_model
            )

            for model_name, model in models.items():
                accuracy = accuracy_score(model.predict(X_test), y_test)
                print(f"{model_name} Accuracy: {accuracy}")

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(predicted, y_test)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
