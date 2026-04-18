# src/components/data_transformation.py

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df):
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = pd.to_datetime(df["date"])

            df["hour"] = df["datetime"].dt.hour
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            df["weekday"] = df["datetime"].dt.weekday

            # Day name
            df["day_of_week"] = df["datetime"].dt.day_name()

            # Weekend flag
            df["is_weekend"] = df["weekday"].apply(
                lambda x: 1 if x in [5, 6] else 0
            )

            # Lag features
            lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]

            for lag in lags:
                df[f"AQI_lag_{lag}"] = df["aqi"].shift(lag)

            # Difference features
            df["AQI_diff_1"] = df["aqi"] - df["AQI_lag_1"]
            df["AQI_diff_24"] = df["aqi"] - df["AQI_lag_24"]

            # Percentage change
            df["AQI_pct_change_1"] = df["aqi"].pct_change(1)
            df["AQI_pct_change_24"] = df["aqi"].pct_change(24)

            # Pollutant lag features
            pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]

            for col in pollutants:
                for lag in [1, 3, 6]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            # Cyclical features
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

            # Interaction features
            df["pm25_no2"] = df["pm25"] * df["no2"]
            df["pm25_o3"] = df["pm25"] * df["o3"]

            # Rolling statistics
            df["AQI_roll_3"] = df["aqi"].rolling(3).mean().shift(1)
            df["AQI_roll_6"] = df["aqi"].rolling(6).mean().shift(1)
            df["AQI_roll_12"] = df["aqi"].rolling(12).mean().shift(1)

            df["AQI_std_3"] = df["aqi"].rolling(3).std().shift(1)
            df["AQI_std_6"] = df["aqi"].rolling(6).std().shift(1)

            # Target
            df["target_AQI"] = df["aqi"].shift(-1)

            df.dropna(inplace=True)

            logging.info("Feature engineering completed")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(
        self,
        numeric_cols,
        categorical_cols
    ):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numeric_cols
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_cols
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded")

            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            target_column = "target_AQI"

            drop_cols = [
                "datetime",
                "date",
                "aqi",
                target_column
            ]

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target_column]

            numeric_cols = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            preprocessor = self.get_data_transformer_object(
                numeric_cols,
                categorical_cols
            )

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Preprocessor saved successfully")

            return (
                X_train_arr,
                y_train,
                X_test_arr,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)