# predict.py

import joblib
import pandas as pd

from feature_eng import load_and_clean, feature_engineering, prepare_target, get_features_targets


# Load model
model = joblib.load("aqi_model.pkl")

# Load data (new or same format)
df = load_and_clean("../data/delhi_ncr_aqi_dataset.csv")

df = feature_engineering(df)
df = prepare_target(df)

X, y = get_features_targets(df)

# Take latest row for prediction
latest = X.iloc[[-1]]

prediction = model.predict(latest)

print("Next AQI Prediction:", prediction[0])