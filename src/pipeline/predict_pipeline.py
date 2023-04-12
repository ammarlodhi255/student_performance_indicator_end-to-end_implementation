import sys
from src.exception import CustomException
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = ModelTrainerConfig().trained_model_path
            preprocessor_path = DataTransformationConfig().preprocessor_path

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            transformed_data = preprocessor.transform(features)
            prediction = model.predict(transformed_data)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self):
        try:
            data = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score]
            }

            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
