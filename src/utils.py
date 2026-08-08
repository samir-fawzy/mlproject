import sys
import os
from src.exception import CustomException
from src.logger import logging
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    """
    Evaluates a dictionary of machine learning models using GridSearchCV.
    Returns a dictionary mapping model names to their test R2 scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"model ({model_name}) in training.")
            para = params.get(model_name, {})

            gs = GridSearchCV(model, para, cv=3, n_jobs=1)
            gs.fit(x_train,y_train)
            models[model_name] = gs.best_estimator_
            logging.info(f"model ({model_name}) finish training successfully.")
            y_test_pred = gs.predict(x_test)

            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"model ({model_name}) score: {test_model_score}")
            report[model_name] = test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)