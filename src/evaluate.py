import os
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)

class ModelEvaluator:

    def __init__(self, model, X_test, y_test, output_dir="output/reports"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_regression(self):
        predictions = self.model.predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        mape = np.mean(
            np.abs((self.y_test - predictions) / (self.y_test + 1e-8))
        ) * 100

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        }

        logging.info(metrics)
        return metrics, predictions

    def cross_validation_scores(self, X_train, y_train):
        scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=5,
            scoring="r2"
        )
        return scores

    def residual_analysis(self, predictions):
        residuals = self.y_test - predictions

        residual_df = pd.DataFrame({
            "Actual": self.y_test,
            "Predicted": predictions,
            "Residuals": residuals
        })

        return residual_df

    def save_metrics(self, metrics):
        metrics_df = pd.DataFrame([metrics])

        path = os.path.join(self.output_dir, "model_metrics.csv")
        metrics_df.to_csv(path, index=False)

        logging.info("Metrics saved")