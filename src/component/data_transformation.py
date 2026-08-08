import sys
import os

import pandas as pd
import numpy as np

# sklearn libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


from dataclasses import dataclass

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformer:

    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        this function is responsible of data transformer
        """
        try:
            num_columns = ["reading_score","writing_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numeric_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns: {cat_columns}")
            logging.info(f"Numerical Columns: {num_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numeric_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformer(self,train_path,test_path):
        """
        this function takes train and test data, do preprocessing and prepare it to feed to ML model\n
        it takes two argument (train path, and test path)\n
        return 3 values (train arr, test arr, preporcessing file)
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            prepocessing_obj=self.get_data_transformer_object()

            logging.info("obtaining preprocessor object")

            target_column_name=['math_score']

            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=prepocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=prepocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            logging.info("saved preprocessing obj")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=prepocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

