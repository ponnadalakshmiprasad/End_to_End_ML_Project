import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.pipelines.exception import CustomException
from src.pipelines.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer   
from src.pipelines.utils import save_object
from src.pipelines.exception import CustomException
from src.pipelines.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Initiate Object for data transformation")

            num_features=["reading_score","writing_score"]
            cat_features=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Completed Object for data transformation")

            logging.info("Initiate pipeline for data transformation")


            logging.info(f"numerical features: {num_features}")
            logging.info(f"categorical features: {cat_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )

            logging.info("Completed pipeline for data transformation")

            logging.info("Saving preprocessor object")

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_data,test_data):
        try:
            logging.info("Initiate data transformation")
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)

            logging.info("reading train and test data completed")

            processing_object=self.get_data_transformation_object()

            target_column="math_score"
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]


            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]


            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying preprocessing on training and testing data")

            train_data_preprocessed=processing_object.fit_transform(input_feature_train_df)
            test_data_preprocessed=processing_object.transform(input_feature_test_df)


            train_data_array=np.c_[train_data_preprocessed,np.array(target_feature_train_df)]
            test_data_array=np.c_[test_data_preprocessed,np.array(target_feature_test_df)]


            logging.info("Completed data transformation")


            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=processing_object
            )

            return (
                train_data_array,
                test_data_array,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            

            


            
            


            
        
