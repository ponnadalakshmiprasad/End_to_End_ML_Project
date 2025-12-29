import sys
import pandas as pd
from src.pipelines.exception import CustomException
from src.pipelines.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            preprocessor=load_object(file_path=preprocessor_path)
            model=load_object(file_path=model_path)
            features = features.fillna("unknown")
            scaled_data=preprocessor.transform(features)
            predictions=model.predict(scaled_data)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        try:
            self.gender=gender
            self.race_ethnicity=race_ethnicity
            self.parental_level_of_education=parental_level_of_education
            self.lunch=lunch
            self.test_preparation_course=test_preparation_course
            self.reading_score=reading_score
            self.writing_score=writing_score
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_as_data_frame(self):
        try:
            data={
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "reading_score":[self.reading_score],
            "writing_score":[self.writing_score]
        }
            data_frame=pd.DataFrame(data)
            return data_frame
        except Exception as e:
            raise CustomException(e,sys)

    
