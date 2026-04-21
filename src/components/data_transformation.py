import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


@dataclass
class DataTransformationConfig:
    scaler_X_path     = os.path.join(PROJECT_ROOT, "artifacts", "scaler_X.pkl")
    scaler_y_path     = os.path.join(PROJECT_ROOT, "artifacts", "scaler_y.pkl")
    preprocessor_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    SEQ_LEN: int      = 48


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df):
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"]     = pd.to_datetime(df["date"])


            df["hour"]        = df["datetime"].dt.hour
            df["day"]         = df["datetime"].dt.day
            df["month"]       = df["datetime"].dt.month
            df["weekday"]     = df["datetime"].dt.weekday
            df["is_weekend"]  = df["weekday"].apply(lambda x: 1 if x in [5, 6] else 0)


            lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
            for lag in lags:
                df[f"AQI_lag_{lag}"] = df["aqi"].shift(lag)

            df["AQI_diff_1"]        = df["aqi"] - df["AQI_lag_1"]
            df["AQI_diff_24"]       = df["aqi"] - df["AQI_lag_24"]
            df["AQI_pct_change_1"]  = df["aqi"].pct_change(1)
            df["AQI_pct_change_24"] = df["aqi"].pct_change(24)


            pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
            for col in pollutants:
                for lag in [1, 3, 6]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]    / 24)
            df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]    / 24)
            df["month_sin"] = np.sin(2 * np.pi * df["month"]   / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"]   / 12)
            df["dow_sin"]   = np.sin(2 * np.pi * df["weekday"] / 7)
            df["dow_cos"]   = np.cos(2 * np.pi * df["weekday"] / 7)

            df["pm25_no2"] = df["pm25"] * df["no2"]
            df["pm25_o3"]  = df["pm25"] * df["o3"]

            df["AQI_roll_3"]  = df["aqi"].rolling(3).mean().shift(1)
            df["AQI_roll_6"]  = df["aqi"].rolling(6).mean().shift(1)
            df["AQI_roll_12"] = df["aqi"].rolling(12).mean().shift(1)
            df["AQI_std_3"]   = df["aqi"].rolling(3).std().shift(1)
            df["AQI_std_6"]   = df["aqi"].rolling(6).std().shift(1)


            df["target_AQI"] = df["aqi"].shift(-1)

            df.dropna(inplace=True)
            logging.info("Feature engineering completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessor(self, numeric_cols, categorical_cols):
        try:
            num_pipeline = Pipeline(steps=[
                ("scaler", MinMaxScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def create_sequences(self, X, y, seq_len):
        try:
            Xs, ys = [], []
            for i in range(seq_len, len(X)):
                Xs.append(X[i - seq_len:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and Test data loaded")

            train_df = self.feature_engineering(train_df)
            test_df  = self.feature_engineering(test_df)

            target_column = "target_AQI"
            drop_cols     = ["datetime", "date", "aqi", target_column]
            drop_cols     = [c for c in drop_cols if c in train_df.columns]

            X_train_df = train_df.drop(columns=drop_cols)
            y_train_df = train_df[target_column]

            X_test_df  = test_df.drop(columns=drop_cols)
            y_test_df  = test_df[target_column]

            numeric_cols     = X_train_df.select_dtypes(
                include=["int64", "float64", "bool"]
            ).columns.tolist()
            categorical_cols = X_train_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            logging.info(f"Numeric cols   : {len(numeric_cols)}")
            logging.info(f"Categorical cols: {categorical_cols}")

            preprocessor   = self.get_preprocessor(numeric_cols, categorical_cols)
            X_train_scaled = preprocessor.fit_transform(X_train_df)
            X_test_scaled  = preprocessor.transform(X_test_df)

            scaler_y       = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train_df.values.reshape(-1, 1))
            y_test_scaled  = scaler_y.transform(y_test_df.values.reshape(-1, 1))

            SEQ_LEN = self.data_transformation_config.SEQ_LEN
            X_train, y_train = self.create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
            X_test,  y_test  = self.create_sequences(X_test_scaled,  y_test_scaled,  SEQ_LEN)

            logging.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

            os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)
            save_object(file_path=self.data_transformation_config.preprocessor_path, obj=preprocessor)
            save_object(file_path=self.data_transformation_config.scaler_y_path,     obj=scaler_y)
            logging.info("Preprocessor and scaler_y saved to artifacts/")

            return (
                X_train,
                y_train,
                X_test,
                y_test,
                self.data_transformation_config.scaler_y_path,
                self.data_transformation_config.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)