import os
import sys
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            # ── 1. Load dataset ──────────────────────────────────────────────
            data_path = os.path.join(PROJECT_ROOT, "Data", "delhi_ncr_aqi_dataset.csv")
            df = pd.read_csv(data_path)
            logging.info("Dataset read successfully")

            # ── 2. Parse datetime columns ────────────────────────────────────
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = pd.to_datetime(df["date"])

            # ── 3. Filter Delhi only ─────────────────────────────────────────
            df = df[df["city"] == "Delhi"].copy()
            df = df.sort_values("datetime").reset_index(drop=True)
            logging.info(f"After filtering Delhi: {df.shape}")

            # ── 4. Drop unwanted columns ─────────────────────────────────────
            df = df.drop(columns=["aqi_category", "station", "city"])
            df.dropna(inplace=True)
            logging.info(f"Dataset shape after cleaning: {df.shape}")

            # ── 5. Save artifacts ────────────────────────────────────────────
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            # ── 6. Time-based train/test split (80/20) ───────────────────────
            split = int(len(df) * 0.8)
            train_set = df.iloc[:split]
            test_set = df.iloc[split:]

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(
                f"Train ({train_set.shape}) and Test ({test_set.shape}) saved to artifacts/"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))