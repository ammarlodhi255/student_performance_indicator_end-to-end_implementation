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

    def get_preprocessor(self):
        '''
        This function creates: 
        1. Two lists (numeric and categorical features).
        2. Two pipelines (numeric and categorical)

        returns: preprocessor of type ColumnTransformer that contains the two pipelines
        for preprocessing the columns of each type. 
        '''

        try:
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

            num_pipeline = Pipeline(
                steps=[
                    # since there are outliers in the data, we use strategy=median.
                    ("Imputer", SimpleImputer(strategy=median)),
                    ("StandardScalar", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy=most_frequent)),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("StandardScalar", StandardScaler())
                ]
            )

            logging.info('Numeric and categorical pipelines created')

            preprocessor = ColumnTransformer(
                [
                    ("Numerical Transformation", num_pipeline, num_features),
                    ("Categorical Transformation", cat_pipeline, cat_features)
                ]
            )

            logging.info('Preprocessor created successfully')
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)