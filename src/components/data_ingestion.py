from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
