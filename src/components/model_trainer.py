import os
import sys
import json
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.keras")
    model_report_path       = os.path.join(PROJECT_ROOT, "artifacts", "model_report.json")
    model_score_path        = os.path.join(PROJECT_ROOT, "artifacts", "model_scores.txt")
    best_model_checkpoint   = os.path.join(PROJECT_ROOT, "artifacts", "best_model.keras")
    tensorboard_log_dir     = os.path.join(PROJECT_ROOT, "logs")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, seq_len, n_features):
        try:
            model = Sequential([

                # ── CNN Block ─────────────────────────────────────────────────
                Conv1D(filters=64, kernel_size=3, activation='relu',
                       padding='same', input_shape=(seq_len, n_features)),
                BatchNormalization(),
                Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),

                # ── Bidirectional LSTM Block ──────────────────────────────────
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.3),

                # ── Output Block ──────────────────────────────────────────────
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)   # Linear — no activation for regression
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.0005),
                loss='mse',
                metrics=['mae']
            )

            logging.info("CNN+BiLSTM model built successfully")
            model.summary(print_fn=logging.info)
            return model

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, y_true, y_pred, scaler_y=None):
        try:
            # Inverse transform back to real AQI range (0-500)
            if scaler_y is not None:
                y_pred  = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                y_true  = scaler_y.inverse_transform(y_true.reshape(-1, 1))

            r2   = r2_score(y_true, y_pred)
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            return r2, mae, rmse

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model Training Started")

            # ── Load scaler_y for inverse transform ───────────────────────────
            scaler_y_path = os.path.join(PROJECT_ROOT, "artifacts", "scaler_y.pkl")
            scaler_y      = load_object(scaler_y_path)

            seq_len    = X_train.shape[1]
            n_features = X_train.shape[2]

            # ── Build model ───────────────────────────────────────────────────
            model = self.build_model(seq_len, n_features)

            # ── Callbacks ─────────────────────────────────────────────────────
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    self.model_trainer_config.best_model_checkpoint,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                ),
                TensorBoard(
                    log_dir=self.model_trainer_config.tensorboard_log_dir
                )
            ]

            # ── Train ─────────────────────────────────────────────────────────
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=1
            )

            logging.info(f"Training stopped at epoch: {len(history.history['loss'])}")

            # ── Evaluate ──────────────────────────────────────────────────────
            y_pred = model.predict(X_test)
            r2, mae, rmse = self.evaluate_model(y_test, y_pred, scaler_y)

            logging.info(f"R2   : {r2:.4f}")
            logging.info(f"MAE  : {mae:.4f}")
            logging.info(f"RMSE : {rmse:.4f}")

            # ── Save report (JSON) ────────────────────────────────────────────
            final_report = {
                "model": "CNN + Bidirectional LSTM",
                "r2_score": float(r2),
                "mae":      float(mae),
                "rmse":     float(rmse),
                "epochs_trained": len(history.history['loss']),
                "seq_len":    seq_len,
                "n_features": n_features
            }

            with open(self.model_trainer_config.model_report_path, "w") as f:
                json.dump(final_report, f, indent=4)

            # ── Save scores (TXT) ─────────────────────────────────────────────
            with open(self.model_trainer_config.model_score_path, "w") as f:
                f.write("MODEL PERFORMANCE REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Model Name     : CNN + Bidirectional LSTM\n")
                f.write(f"R2 Score       : {r2:.4f}\n")
                f.write(f"MAE            : {mae:.4f}\n")
                f.write(f"RMSE           : {rmse:.4f}\n")
                f.write(f"Epochs Trained : {len(history.history['loss'])}\n")
                f.write(f"Seq Length     : {seq_len}\n")
                f.write(f"Features       : {n_features}\n")
                f.write("-" * 60 + "\n")

            # ── Save final model ──────────────────────────────────────────────
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info("Model saved successfully to artifacts/model.keras")

            return r2

        except Exception as e:
            raise CustomException(e, sys)