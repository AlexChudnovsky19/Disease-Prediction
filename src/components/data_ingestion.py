import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str

class DataIngestion:
    def __init__(self, train_data_path, test_data_path, raw_data_path):
        self.ingestion_config = DataIngestionConfig(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            raw_data_path=raw_data_path
        )

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Read the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

# Usage
if __name__ == "__main__":
    csv_files = [ r'\Users\alexc\Heart_failure_predictor\notebook\Heart_Failure_predict.csv']
                 #r'\Users\alexc\Heart_failure_predictor\notebook\Diabetes_Predict.csv']
                 #r'\Users\alexc\Heart_failure_predictor\notebook\Stroke_predict.csv']


    for csv_file in csv_files:
        train_path = os.path.join("artifacts", f"train_{os.path.basename(csv_file)}")
        test_path = os.path.join("artifacts", f"test_{os.path.basename(csv_file)}")

        obj = DataIngestion(train_path, test_path, csv_file)
        train_data,test_data = obj.initiate_data_ingestion()

