import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor():

        num_features = [
            'reading score',
            'writing score'
        ]

        cat_features = [
            'gender',
            'parental level of education',
            'lunch',
            'race/ethnicity',
            'test preparation course'
        ]
