import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import scipy
from scipy import sparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            text_column = "full_text"
            categorical_columns = ["department"]
            numerical_columns = [
                "has_salary", "has_company_logo", 
                "telecommuting", "has_questions", 
                "text_length", "word_count"
            ]

            text_pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english"))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("text_pipeline", text_pipeline, text_column),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ("num_pipeline", "passthrough", numerical_columns)
                ],
                remainder="drop" 
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "fraudulent"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = scipy.sparse.hstack((input_feature_train_arr, target_feature_train_df.values.reshape(-1,1)))
            test_arr = scipy.sparse.hstack((input_feature_test_arr, target_feature_test_df.values.reshape(-1,1)))

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)