import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info('Splitting data into X and Y')

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "RandomForestRegressor": RandomForestRegressor()
            }

            logging.info('all models initialized')

            evaluation_report: dict = evaluate_models(
                self, X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(evaluation_report.values()))
            best_model_name = list(evaluation_report.keys())[list(evaluation_report.values()).index(best_model_score)]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found!')

            model = models[best_model_name]
            logging.info('Best model is found. Saving...')
            save_object(model, self.model_trainer_config.trained_model_path)
            logging.info('Model saved!')

        except Exception as e:
            raise CustomException(e, sys)
