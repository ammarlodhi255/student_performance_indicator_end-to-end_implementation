import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
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
                    ("Imputer", SimpleImputer(strategy='median')),
                    ("StandardScalar", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("StandardScalar", StandardScaler(with_mean=False))
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

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function performs data transformation on the inputs train and test data.

        Returns: 
            train_arr: transformed array of train set.
            test_arr: transformed array of test set.
            preprocessor_obj: preprocessor object used to transform the data.
        '''

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_preprocessor()
            logging.info('Preprocessor is loaded')
            target_feature = 'math score'

            input_train_df = train_df.drop(columns=[target_feature], axis=1)
            input_test_df = test_df.drop(columns=[target_feature], axis=1)

            output_train_df = train_df[target_feature]
            output_test_df = test_df[target_feature]

            train_arr = preprocessor_obj.fit_transform(input_train_df)
            test_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[train_arr, np.array(output_train_df)]
            test_arr = np.c_[test_arr, np.array(output_test_df)]

            logging.info('Data has been transformed')
            logging.info('Saving preprocessor object')

            save_object(
                obj=preprocessor_obj,
                file_path=self.data_transformation_config.preprocessor_path
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)
