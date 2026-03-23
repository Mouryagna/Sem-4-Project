# train.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from feature_eng import load_and_clean, feature_engineering, prepare_target, get_features_targets


# Load + process
df = load_and_clean("../data/delhi_ncr_aqi_dataset.csv")
df = feature_engineering(df)
df = prepare_target(df)

X, y = get_features_targets(df)

# Time split
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# Save model
joblib.dump(model, "aqi_model.pkl")