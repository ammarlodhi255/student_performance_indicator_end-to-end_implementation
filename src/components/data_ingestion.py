from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion():

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        This function initiates the data ingestion and returns the path to train, 
        test, and raw data.

        returns: Tuple of three paths (train, test, and raw data)
        '''

        logging.info('Initiating the data ingestion')

        try:
            # Read data from the source (database, local files, etc)
            data = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Data is read as a dataframe')

            # Creating directories
            logging.info('Create artifacts directory and save the raw data')
            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,
                        index=False, header=True)

            # Split the data into train and test
            logging.info('Train test split initiated')
            train, test = train_test_split(data, test_size=.2, random_state=42)

            logging.info('Saving train test split inside artifacts directory')
            train.to_csv(self.ingestion_config.train_data_path,
                         index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path,
                        index=False, header=True)

            logging.info('Ingestion is complete')

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info(f'Exception is raised: {e}')
            raise CustomException(e, sys)


if __name__ == "__main__":
    di = DataIngestionConfig()
    dt = DataTransformation()
    train, test, preprocessor_path = dt.initiate_data_transformation(
        di.train_data_path, di.test_data_path)
    mt = ModelTrainer()
    print(mt.initiate_model_training(train, test))
