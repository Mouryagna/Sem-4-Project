from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

import sys


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training Pipeline Started")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion Completed")

            transformation = DataTransformation()
            X_train, y_train, X_test, y_test, scaler_y_path, preprocessor_path = \
                transformation.initiate_data_transformation(train_path, test_path)
            logging.info("Data Transformation Completed")

            trainer = ModelTrainer()
            best_score = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
            logging.info(f"Model Training Completed. R2 Score: {best_score:.4f}")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()