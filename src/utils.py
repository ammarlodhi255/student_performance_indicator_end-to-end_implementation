import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(obj, file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(self, X_train, y_train, X_test, y_test, models):
    try:
        evaluation_report = {}

        for key in models.keys():
            model = models[key]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_train_score = r2_score(y_train, y_train_pred)

            y_test_pred = model.predict(X_test)
            y_test_score = r2_score(y_test, y_test_pred)

            evaluation_report[key] = y_test_score

        return evaluation_report
    except Exception as e:
        raise CustomException(e, sys)
