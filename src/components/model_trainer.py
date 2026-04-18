import os
import sys
import json
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "model.pkl"
    )

    model_report_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "model_report.json"
    )

    model_score_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "model_scores.txt"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return r2, mae, rmse

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model Training Started")

            models = {
                "Linear Regression": LinearRegression(),

                "Random Forest": RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

                "XGBoost": XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=42
                ),

                "CatBoost": CatBoostRegressor(
                    iterations=300,
                    learning_rate=0.05,
                    depth=8,
                    verbose=0,
                    random_state=42
                ),

                "LightGBM": LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=42
                )
            }

            model_report = {}

            best_model_score = -999
            best_model_name = None
            best_model = None

            for model_name, model in models.items():

                logging.info(f"Training Started : {model_name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                r2, mae, rmse = self.evaluate_model(
                    y_test,
                    y_pred
                )

                model_report[model_name] = {
                    "r2_score": float(r2),
                    "mae": float(mae),
                    "rmse": float(rmse)
                }

                logging.info(f"{model_name} R2 Score : {r2}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model_name = model_name
                    best_model = model

            final_report = {
                "best_model_name": best_model_name,
                "best_model_score": float(best_model_score),
                "all_models": model_report
            }

            with open(
                self.model_trainer_config.model_report_path,
                "w"
            ) as f:
                json.dump(final_report, f, indent=4)

            with open(
                self.model_trainer_config.model_score_path,
                "w"
            ) as f:

                f.write("MODEL PERFORMANCE REPORT\n")
                f.write("=" * 60 + "\n\n")

                for model_name, scores in model_report.items():

                    f.write(f"Model Name : {model_name}\n")
                    f.write(
                        f"R2 Score   : {scores['r2_score']:.4f}\n"
                    )
                    f.write(
                        f"MAE        : {scores['mae']:.4f}\n"
                    )
                    f.write(
                        f"RMSE       : {scores['rmse']:.4f}\n"
                    )
                    f.write("-" * 60 + "\n")

                f.write("\n")
                f.write(
                    f"Best Model : {best_model_name}\n"
                )
                f.write(
                    f"Best Score : {best_model_score:.4f}\n"
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best Model Saved Successfully")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)