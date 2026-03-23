# feature_eng.py

import pandas as pd
import numpy as np


def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("datetime").reset_index(drop=True)

    # Filter Delhi
    df = df[df["city"] == "Delhi"].copy()

    # Drop useless columns
    df = df.drop(columns=["aqi_category", "station", "city"])

    return df


def feature_engineering(df):
    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    # AQI lags
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        df[f"AQI_lag_{lag}"] = df["aqi"].shift(lag)

    # AQI diff + pct
    df["AQI_diff_1"] = df["aqi"] - df["AQI_lag_1"]
    df["AQI_diff_24"] = df["aqi"] - df["AQI_lag_24"]

    df["AQI_pct_change_1"] = df["aqi"].pct_change(1)
    df["AQI_pct_change_24"] = df["aqi"].pct_change(24)

    # Pollutant lags
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]

    for col in pollutants:
        for lag in [1, 3, 6]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Interaction
    df["pm25_no2"] = df["pm25"] * df["no2"]
    df["pm25_o3"] = df["pm25"] * df["o3"]

    # Rolling stats
    df["AQI_roll_3"] = df["aqi"].rolling(3).mean().shift(1)
    df["AQI_roll_6"] = df["aqi"].rolling(6).mean().shift(1)
    df["AQI_roll_12"] = df["aqi"].rolling(12).mean().shift(1)

    df["AQI_std_3"] = df["aqi"].rolling(3).std().shift(1)
    df["AQI_std_6"] = df["aqi"].rolling(6).std().shift(1)

    # Encode season
    df = pd.get_dummies(df, columns=["season"], drop_first=True)

    return df


def prepare_target(df):
    df["target_AQI"] = df["aqi"].shift(-1)
    df = df.dropna()
    return df


def get_features_targets(df):
    X = df.drop(columns=["datetime", "aqi", "target_AQI", "date", "day_of_week"])
    y = df["target_AQI"]
    return X, y