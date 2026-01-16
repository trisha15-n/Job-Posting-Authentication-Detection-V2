import os
import sys
from dataclasses import dataclass
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            if hasattr(train_array, 'tocsr'):
                train_array = train_array.tocsr()
            if hasattr(test_array, 'tocsr'):
                test_array = test_array.tocsr()

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            try:
                y_train = y_train.toarray().ravel()
                y_test = y_test.toarray().ravel()
            except:
                y_train = y_train.ravel()
                y_test = y_test.ravel()

            model = RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=42,
                n_jobs=-1
            )

            logging.info("model training")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logging.info(f"Accuracy: {acc:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(classification_report(y_test, y_pred))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            return acc

        except Exception as e:
            raise CustomException(e, sys)