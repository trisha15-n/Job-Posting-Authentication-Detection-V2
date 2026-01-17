import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Loading Model and Preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("Transforming Input Data...")
            data_scaled = preprocessor.transform(features)
            
            print("Predicting...")
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:

    def __init__(self,
        title: str,
        company_profile: str,
        description: str,
        requirements: str,
        benefits: str,
        telecommuting: int,
        has_company_logo: int,
        has_questions: int,
        salary_range: str,
        department: str
    ):
        self.title = title
        self.company_profile = company_profile
        self.description = description
        self.requirements = requirements
        self.benefits = benefits
        self.telecommuting = telecommuting
        self.has_company_logo = has_company_logo
        self.has_questions = has_questions
        self.salary_range = salary_range
        self.department = department

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "title": [self.title],
                "company_profile": [self.company_profile],
                "description": [self.description],
                "requirements": [self.requirements],
                "benefits": [self.benefits],
                "telecommuting": [self.telecommuting],
                "has_company_logo": [self.has_company_logo],
                "has_questions": [self.has_questions],
                "salary_range": [self.salary_range],
                "department": [self.department]
            }

            df = pd.read_json(pd.DataFrame(custom_data_input_dict).to_json())

            text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
            df[text_cols] = df[text_cols].fillna(" ")
            
            df['full_text'] = (
                df['title'] + " " + df['company_profile'] + " " + 
                df['description'] + " " + df['requirements'] + " " + 
                df['benefits']
            )

            df['has_salary'] = df['salary_range'].apply(lambda x: 0 if pd.isna(x) or x == "" else 1)
            
            df['text_length'] = df['full_text'].apply(len)
            df['word_count'] = df['full_text'].apply(lambda x: len(x.split()))

            df['department'] = df['department'].astype(str).str.lower().str.strip()
            final_cols = [
                'full_text', 'department', 
                'has_salary', 'has_company_logo', 
                'telecommuting', 'has_questions', 
                'text_length', 'word_count'
            ]
            
            return df[final_cols]

        except Exception as e:
            raise CustomException(e, sys)