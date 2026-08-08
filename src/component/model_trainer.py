import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def _get_models_and_params(self):
        models={
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(verbosity=0), # إيقاف التحذيرات المزعجة
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }

        params = {
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},
            "XGBRegressor": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoosting Regressor": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor": {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }
        return models, params

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split data -> (X,Y)")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("data has been splited successfully.")

            models, params = self._get_models_and_params()
            logging.info("models are training")
            models_report:dict=evaluate_models(
                x_train, y_train,
                x_test, y_test,
                models, params
            )

            best_model_name = max(models_report, key=models_report.get)
            best_model_score = models_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No acceptable model found. Best score is below 0.6", sys)

            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e,sys)


