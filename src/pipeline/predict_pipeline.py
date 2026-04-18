# src/pipeline/predict_pipeline.py

import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def create_backend_features(self, df):
        try:
            # ---------------- Time Features ----------------
            df["day_of_week"] = df["weekday"].map({
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday"
            })

            df["is_weekend"] = df["weekday"].apply(
                lambda x: 1 if x in [5, 6] else 0
            )

            # ---------------- Cyclical Features ----------------
            df["hour_sin"] = np.sin(
                2 * np.pi * df["hour"] / 24
            )

            df["hour_cos"] = np.cos(
                2 * np.pi * df["hour"] / 24
            )

            df["month_sin"] = np.sin(
                2 * np.pi * df["month"] / 12
            )

            df["month_cos"] = np.cos(
                2 * np.pi * df["month"] / 12
            )

            df["dow_sin"] = np.sin(
                2 * np.pi * df["weekday"] / 7
            )

            df["dow_cos"] = np.cos(
                2 * np.pi * df["weekday"] / 7
            )

            # ---------------- Interaction Features ----------------
            df["pm25_no2"] = df["pm25"] * df["no2"]
            df["pm25_o3"] = df["pm25"] * df["o3"]

            # ---------------- Historical Raw Data ----------------
            hist = pd.read_csv("artifacts/raw.csv")

            hist["datetime"] = pd.to_datetime(
                hist["datetime"]
            )

            hist = hist.sort_values(
                "datetime"
            ).reset_index(drop=True)

            aqi_series = hist["aqi"].tail(200).tolist()

            current_proxy_aqi = (
                df["pm25"].iloc[0] * 0.5 +
                df["pm10"].iloc[0] * 0.3 +
                df["no2"].iloc[0] * 0.2
            )

            aqi_series.append(current_proxy_aqi)

            s = pd.Series(aqi_series)

            # ---------------- AQI Lag Features ----------------
            lag_values = [1, 2, 3, 6, 12, 24, 48, 72, 168]

            for lag in lag_values:
                df[f"AQI_lag_{lag}"] = s.shift(
                    lag
                ).iloc[-1]

            # ---------------- AQI Diff ----------------
            df["AQI_diff_1"] = (
                df["AQI_lag_1"] -
                df["AQI_lag_2"]
            )

            df["AQI_diff_24"] = (
                df["AQI_lag_1"] -
                df["AQI_lag_24"]
            )

            # ---------------- AQI Percentage Change ----------------
            df["AQI_pct_change_1"] = (
                (
                    df["AQI_lag_1"] -
                    df["AQI_lag_2"]
                ) /
                (df["AQI_lag_2"] + 1e-5)
            )

            df["AQI_pct_change_24"] = (
                (
                    df["AQI_lag_1"] -
                    df["AQI_lag_24"]
                ) /
                (df["AQI_lag_24"] + 1e-5)
            )

            # ---------------- Rolling Features ----------------
            df["AQI_roll_3"] = s.tail(3).mean()
            df["AQI_roll_6"] = s.tail(6).mean()
            df["AQI_roll_12"] = s.tail(12).mean()

            df["AQI_std_3"] = s.tail(3).std()
            df["AQI_std_6"] = s.tail(6).std()

            # ---------------- Pollutant Lag Features ----------------
            pollutant_cols = [
                "pm25",
                "pm10",
                "no2",
                "so2",
                "co",
                "o3"
            ]

            for col in pollutant_cols:
                val = df[col].iloc[0]

                df[f"{col}_lag_1"] = val
                df[f"{col}_lag_3"] = val
                df[f"{col}_lag_6"] = val

            # ---------------- Static Columns ----------------
            df["temperature"] = 30
            df["humidity"] = 60
            df["wind_speed"] = 8
            df["visibility"] = 4
            df["latitude"] = 28.6139
            df["longitude"] = 77.2090
            df["year"] = 2026

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            final_df = self.create_backend_features(
                features
            )

            transformed_data = preprocessor.transform(
                final_df
            )

            prediction = model.predict(
                transformed_data
            )

            logging.info(
                f"Prediction completed: {prediction}"
            )

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        pm25,
        pm10,
        no2,
        so2,
        co,
        o3,
        hour,
        day,
        month,
        weekday,
        season
    ):
        self.pm25 = pm25
        self.pm10 = pm10
        self.no2 = no2
        self.so2 = so2
        self.co = co
        self.o3 = o3
        self.hour = hour
        self.day = day
        self.month = month
        self.weekday = weekday
        self.season = season

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "pm25": [self.pm25],
                "pm10": [self.pm10],
                "no2": [self.no2],
                "so2": [self.so2],
                "co": [self.co],
                "o3": [self.o3],
                "hour": [self.hour],
                "day": [self.day],
                "month": [self.month],
                "weekday": [self.weekday],
                "season": [self.season]
            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:
            raise CustomException(e, sys)