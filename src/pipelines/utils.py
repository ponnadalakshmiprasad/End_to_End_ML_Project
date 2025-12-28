import os
import dill
import sys
import numpy as np
import pandas as pd
from src.pipelines.exception import CustomException
from src.pipelines.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        logging.info("Enter the save_object method of utils")
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        model_list={}

        for i in range(len(list(models.values()))):
            model=list(models.values())[i]
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            r2_score_train=r2_score(y_train,y_train_pred)
            r2_score_test=r2_score(y_test,y_test_pred)
            model_list.update({list(models.keys())[i]:r2_score_test})

        return model_list

    except Exception as e:
        raise CustomException(e,sys)