import os
import sys
import pandas as pd
import numpy as np
from src.pipelines.exception import CustomException
from src.pipelines.logger import logging
from src.pipelines.utils import save_object, evaluate_models
from dataclasses import dataclass
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
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

            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0]
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1.0]
                },
                "SVR": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf"]
                },
                "KNeighborsRegressor": {
                    "n_neighbors": [3,5,7,9]
                },

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            models_data:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

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
            print(best_model)
            return r2_score(y_test,y_pred)*100


        except Exception as e:
            raise CustomException(e,sys)

            










            




            