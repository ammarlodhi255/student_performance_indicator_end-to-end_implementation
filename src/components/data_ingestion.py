from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join('')
