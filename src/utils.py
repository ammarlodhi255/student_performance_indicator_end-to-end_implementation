import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException


def save_object(obj, file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
