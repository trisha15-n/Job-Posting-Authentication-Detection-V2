import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            original_data_path = Path("data/fake_job_postings.csv")
            
            if not original_data_path.exists():
                raise FileNotFoundError(f"Raw data not found at {original_data_path}")

            df = pd.read_csv(original_data_path)
            logging.info(f"Read dataset as dataframe. Shape: {df.shape}")

            text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
            df[text_cols] = df[text_cols].fillna(" ")
            
            df['full_text'] = (
                df['title'] + " " + 
                df['company_profile'] + " " + 
                df['description'] + " " + 
                df['requirements'] + " " + 
                df['benefits']
            )
            df['has_company_logo'] = df['has_company_logo'].fillna(0).astype(int)
            df['has_salary'] = df['salary_range'].notna().astype(int)
            df['department'] = df['department'].astype(str).str.lower().str.strip()
            df['department'] = df['department'].replace('nan', 'unknown')
            dept_counts = df['department'].value_counts()
            rare_depts = dept_counts[dept_counts < 50].index
            df['department'] = df['department'].apply(lambda x: 'other' if x in rare_depts else x)
          
            df['text_length'] = df['full_text'].apply(len)
            df['word_count'] = df['full_text'].apply(lambda x: len(x.split()))
            
            final_cols = [
                'fraudulent', 'full_text', 'department', 
                'has_salary', 'has_company_logo', 
                'telecommuting', 'has_questions', 
                'text_length', 'word_count'
            ]
            df['telecommuting'] = df['telecommuting'].fillna(0).astype(int)
            df['has_questions'] = df['has_questions'].fillna(0).astype(int)
            
            df = df[final_cols]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("splitting data into train and test")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion COmpleted.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Data Ingestion Complete.\nTrain Data: {train_data}\nTest Data: {test_data}")