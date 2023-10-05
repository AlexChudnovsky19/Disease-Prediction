import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer , make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from Model_Trainer import ModelTrainer
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    def __init__(self, disease_name):
        self.preprocessor_obj_file_path = os.path.join('artifacts', f"preprocessor_{disease_name}.pkl")



class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        self.one_hot_encoder.fit(X)
        return self

    def transform(self, X):
        return self.one_hot_encoder.transform(X)

class DataTransformation:
    def __init__(self, disease_name):
        self.data_transformation_config = DataTransformationConfig(disease_name)

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", CustomOneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, target_column_name):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = train_df.select_dtypes(include=[object]).columns.tolist()

            numerical_columns.remove(target_column_name)

            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_df = pd.DataFrame(input_feature_train_df)
            input_feature_test_df = pd.DataFrame(input_feature_test_df)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")



            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
           

            return (
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e, sys)

# ...

if __name__ == "__main__":
    disease_info = [
        #{
        #    "csv_path": r'\Users\alexc\Heart_failure_predictor\notebook\Heart_Failure_predict.csv',
        #    "target_column": 'HeartDisease',
        #    "name" : 'Heart_Failure'
        #},
        {
            "csv_path": r'\Users\alexc\Heart_failure_predictor\notebook\Diabetes_predict.csv',
            "target_column": 'diabetes',
            "name" : 'Diabetes'
        },
        #{
        #    "csv_path": r'\Users\alexc\Heart_failure_predictor\notebook\Stroke_predict.csv',
        #    "target_column": 'stroke',
        #    "name" : 'Stroke'
        #},
        # Add more disease information as needed
    ]

    for disease_data in disease_info:
        train_path = os.path.join("artifacts", f"train_{os.path.basename(disease_data['csv_path'])}")
        test_path = os.path.join("artifacts", f"test_{os.path.basename(disease_data['csv_path'])}")

        # Create a DataTransformation instance and pass the preprocessor path
        data_transformation = DataTransformation(disease_data['name'])
        train_arr, test_arr = data_transformation.initiate_data_transformation(
            train_path, test_path, disease_data['target_column']
        )


        # Create a ModelTrainer instance
        model_trainer = ModelTrainer()

        print(f"Prediction of {disease_data['target_column']} started:")
        
        # Call the initiate_model_trainer method with disease-specific data and preprocessor_path
        accuracy = model_trainer.initiate_model_trainer(disease_data['name'], train_arr, test_arr)
        
        print(f"Accuracy for {disease_data['target_column']}: {accuracy}")
