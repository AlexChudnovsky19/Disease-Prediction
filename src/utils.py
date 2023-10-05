import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,X_test,y_train,y_test,models,param):
    try:
        report = {}

        for i, (model_name, model) in enumerate(models.items()):
           
            para=param[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3,random_state=0)
            gs.fit(X_train, y_train)
                
            
            model.set_params(**gs.best_params_)

            print(f"utils {X_train.shape} ")
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate model scores (e.g., R2 score)
            #train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            test_model_accuracy = accuracy_score(y_test_pred , y_test) 
            print(f"{test_model_accuracy}")
            report[model_name] = test_model_accuracy

        return report


    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        src_directory = os.path.abspath('C:\\Users\\alexc\\Heart_failure_predictor\\src\\components')
        sys.path.append(src_directory)
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)