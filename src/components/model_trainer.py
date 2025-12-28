import os
import sys
import pandas as pd
import numpy as np
from src.pipelines.exception import CustomException
from src.pipelines.logger import logging
from src.pipelines.utils import save_object, evaluate_models
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig:
    model_obj_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Initiate model training")

            x_train,y_train=train_array[:,:-1],train_array[:,-1]
            x_test,y_test=test_array[:,:-1],test_array[:,-1]

            models={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "SVR":SVR(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()
            }


            models_data:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score=max(sorted(models_data.values()))
            best_model_name=list(models_data.keys())[list(models_data.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise Exception("No best model found")
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.model_obj_file_path,
                obj=best_model
                )
            
            best_model.fit(x_train,y_train)
            y_pred=best_model.predict(x_test)
            return r2_score(y_test,y_pred)*100

        except Exception as e:
            raise CustomException(e,sys)










            




            