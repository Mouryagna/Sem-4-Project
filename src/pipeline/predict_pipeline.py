import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

SEQ_LEN = 48  # must match training SEQ_LEN


class PredictPipeline:
    def __init__(self):
        pass

    def create_backend_features(self, df, hist):
        """
        Build all engineered features for a single input row.
        hist  → last 200+ rows from raw.csv (already loaded & sorted)
        df    → single row DataFrame from CustomData
        """
        try:
            # ── Time features ────────────────────────────────────────────────
            df["day_of_week"] = df["weekday"].map({
                0: "Monday",    1: "Tuesday",  2: "Wednesday",
                3: "Thursday",  4: "Friday",   5: "Saturday",
                6: "Sunday"
            })
            df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [5, 6] else 0)

            # ── Cyclical features ────────────────────────────────────────────
            df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]    / 24)
            df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]    / 24)
            df["month_sin"] = np.sin(2 * np.pi * df["month"]   / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"]   / 12)
            df["dow_sin"]   = np.sin(2 * np.pi * df["weekday"] / 7)
            df["dow_cos"]   = np.cos(2 * np.pi * df["weekday"] / 7)

            # ── Interaction features ─────────────────────────────────────────
            df["pm25_no2"] = df["pm25"] * df["no2"]
            df["pm25_o3"]  = df["pm25"] * df["o3"]

            # ── Build AQI series from history + current proxy ─────────────────
            aqi_series = hist["aqi"].tail(200).tolist()
            current_proxy_aqi = (
                df["pm25"].iloc[0] * 0.5 +
                df["pm10"].iloc[0] * 0.3 +
                df["no2"].iloc[0]  * 0.2
            )
            aqi_series.append(current_proxy_aqi)
            s = pd.Series(aqi_series)

            # ── AQI lag features ─────────────────────────────────────────────
            for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
                df[f"AQI_lag_{lag}"] = s.shift(lag).iloc[-1]

            # ── AQI diff & pct change ────────────────────────────────────────
            df["AQI_diff_1"]        = df["AQI_lag_1"] - df["AQI_lag_2"]
            df["AQI_diff_24"]       = df["AQI_lag_1"] - df["AQI_lag_24"]
            df["AQI_pct_change_1"]  = (df["AQI_lag_1"] - df["AQI_lag_2"])  / (df["AQI_lag_2"]  + 1e-5)
            df["AQI_pct_change_24"] = (df["AQI_lag_1"] - df["AQI_lag_24"]) / (df["AQI_lag_24"] + 1e-5)

            # ── Rolling statistics ───────────────────────────────────────────
            df["AQI_roll_3"]  = s.tail(3).mean()
            df["AQI_roll_6"]  = s.tail(6).mean()
            df["AQI_roll_12"] = s.tail(12).mean()
            df["AQI_std_3"]   = s.tail(3).std()
            df["AQI_std_6"]   = s.tail(6).std()

            # ── Pollutant lag features ───────────────────────────────────────
            for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                val = df[col].iloc[0]
                df[f"{col}_lag_1"] = val
                df[f"{col}_lag_3"] = val
                df[f"{col}_lag_6"] = val

            # ── Static columns — exist in dataset, use historical averages ──
            if "temperature" not in df.columns:
                df["temperature"] = hist["temperature"].mean() if "temperature" in hist.columns else 25.0
            if "humidity" not in df.columns:
                df["humidity"]    = hist["humidity"].mean()    if "humidity"    in hist.columns else 60.0
            if "wind_speed" not in df.columns:
                df["wind_speed"]  = hist["wind_speed"].mean()  if "wind_speed"  in hist.columns else 8.0
            if "visibility" not in df.columns:
                df["visibility"]  = hist["visibility"].mean()  if "visibility"  in hist.columns else 4.0
            if "latitude" not in df.columns:
                df["latitude"]    = 28.6139
            if "longitude" not in df.columns:
                df["longitude"]   = 77.2090
            if "year" not in df.columns:
                df["year"]        = pd.Timestamp.now().year

            # season      kept as raw string — OHE in preprocessor handles it
            # day_of_week kept as raw string — OHE in preprocessor handles it

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def build_sequence(self, current_row_df, preprocessor, hist):
        """
        LSTM needs (1, SEQ_LEN, n_features) not just (1, n_features).
        Strategy: take last SEQ_LEN-1 rows from raw.csv history,
        engineer features for each, then append the current row.
        This gives a proper (1, 48, n_features) sequence.
        """
        try:

            window = hist.tail(SEQ_LEN - 1).copy()

            rows = []

            for _, row in window.iterrows():
                row_df = pd.DataFrame([{
                    "pm25":    row["pm25"],
                    "pm10":    row["pm10"],
                    "no2":     row["no2"],
                    "so2":     row["so2"],
                    "co":      row["co"],
                    "o3":      row["o3"],
                    "hour":    pd.to_datetime(row["datetime"]).hour,
                    "day":     pd.to_datetime(row["datetime"]).day,
                    "month":   pd.to_datetime(row["datetime"]).month,
                    "weekday": pd.to_datetime(row["datetime"]).weekday(),
                    "season":  row.get("season", "Winter")
                }])
                row_df = self.create_backend_features(row_df, hist)
                rows.append(row_df)


            rows.append(current_row_df)

            sequence_df = pd.concat(rows, ignore_index=True)


            sequence_scaled = preprocessor.transform(sequence_df)


            sequence_3d = sequence_scaled.reshape(1, SEQ_LEN, -1)

            return sequence_3d

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            model_path        = os.path.join(PROJECT_ROOT, "artifacts", "model.keras")
            preprocessor_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
            scaler_y_path     = os.path.join(PROJECT_ROOT, "artifacts", "scaler_y.pkl")

            model        = tf.keras.models.load_model(model_path)
            preprocessor = load_object(preprocessor_path)
            scaler_y     = load_object(scaler_y_path)


            raw_path = os.path.join(PROJECT_ROOT, "artifacts", "raw.csv")
            hist = pd.read_csv(raw_path)
            hist["datetime"] = pd.to_datetime(hist["datetime"])
            hist = hist.sort_values("datetime").reset_index(drop=True)


            final_df = self.create_backend_features(features.copy(), hist)


            sequence_3d = self.build_sequence(final_df, preprocessor, hist)


            pred_scaled = model.predict(sequence_3d)


            pred_actual = scaler_y.inverse_transform(pred_scaled)

            logging.info(f"Prediction completed: {pred_actual[0][0]:.2f}")

            return round(float(pred_actual[0][0]), 2)

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        pm25, pm10, no2, so2, co, o3,
        hour, day, month, weekday, season
    ):
        self.pm25    = pm25
        self.pm10    = pm10
        self.no2     = no2
        self.so2     = so2
        self.co      = co
        self.o3      = o3
        self.hour    = hour
        self.day     = day
        self.month   = month
        self.weekday = weekday
        self.season  = season

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "pm25":    [self.pm25],
                "pm10":    [self.pm10],
                "no2":     [self.no2],
                "so2":     [self.so2],
                "co":      [self.co],
                "o3":      [self.o3],
                "hour":    [self.hour],
                "day":     [self.day],
                "month":   [self.month],
                "weekday": [self.weekday],
                "season":  [self.season]
            })
        except Exception as e:
            raise CustomException(e, sys)