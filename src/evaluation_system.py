import os
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)


class EvaluationSystem:

    def __init__(self, output_dir="output/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # REGRESSION METRICS
    # ---------------------------------------------------------
    def evaluate_regression(self, model, X_test, y_test):

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    # ---------------------------------------------------------
    # CLASSIFICATION METRICS
    # ---------------------------------------------------------
    def evaluate_classification(self, model, X_test, y_test):

        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)

        return {
            "AUC": auc
        }

    # ---------------------------------------------------------
    # CROSS VALIDATION
    # ---------------------------------------------------------
    def cross_validate(self, model, X_train, y_train, task="regression"):

        scoring = "r2" if task == "regression" else "roc_auc"

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=scoring
        )

        return scores.mean()

    # ---------------------------------------------------------
    # MODEL COMPARISON
    # ---------------------------------------------------------
    def compare_models(self, models_dict, X_train, X_test, y_train, y_test):

        results = []

        for name, model in models_dict.items():

            model.fit(X_train, y_train)

            metrics = self.evaluate_regression(model, X_test, y_test)
            cv_score = self.cross_validate(model, X_train, y_train)

            metrics["Model"] = name
            metrics["CV_Score"] = cv_score

            results.append(metrics)

        df = pd.DataFrame(results)

        # Sort by RMSE (lower is better)
        df = df.sort_values(by="RMSE")

        return df

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------
    def save_report(self, df, filename="model_comparison.csv"):

        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)

        logging.info("Model comparison saved")

        return path