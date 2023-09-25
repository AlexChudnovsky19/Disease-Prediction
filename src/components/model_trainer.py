import os
import sys
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from Logistic_Regression import LogisticRegression
from Naive_Bayes import NaiveBayesClassifier
from Decision_Trees import DecisionTree
from Linear_Regression import LinearRegression
from K_Nearest_Neighbors import KNearestNeighbors
from Support_Vector_Machines import Support_Vector_Machine
from Random_Forest import RandomForest
from Neural_Network import NeuralNetwork
from src.utils import save_object,evaluate_models
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                #"Logistic Regression" : LogisticRegression(lr=0.05, max_iters=40000, tol=0.005),
                #"Gaussian Naive Bayes" : NaiveBayesClassifier(),
                #"Decision Trees" : DecisionTree(),
                #"K-Nearest Neighbors": KNearestNeighbors(k=20),
                #"Support Vector Machines" : Support_Vector_Machine(kernel='rbf', C=1.0, gamma='scale', random_state=None),
                #"Random Forest" : RandomForest(),
                "Neural Network" : NeuralNetwork(shape=X_train.shape[0],hidden_units=10,alpha = 0.1 ,iterations=1000)

            }
            params={
                #"Logistic Regression":{
                    #'lr': [0.05,0.01],
                    #'max_iters': [40000,1000],
                    #'tol': [0.005,1e-3]
                #},
                #"Gaussian Naive Bayes": {},
                #"Decision Trees" : {},
                #"K-Nearest Neighbors": {},
                #"Support Vector Machines": {},
                #"Random Forest" : {},
                "Neural Network" :{}
                
            }
            print(f"before evaluate{X_train.shape}")
            model_report:dict=evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
                                            models=models,param=params)
            
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            for model_name, model in models.items():
              accuracy = accuracy_score(model.predict(X_test), y_test)
              print(f"{model_name} Accuracy: {accuracy}")

            predicted=best_model.predict(X_test)

            #r2_square = r2_score(y_test, predicted)
            accuracy = accuracy_score(predicted , y_test)

            return accuracy
            
   
        except Exception as e:
            raise CustomException(e,sys)