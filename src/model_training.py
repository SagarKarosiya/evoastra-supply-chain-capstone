import os
import logging
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from keras.layers import Dense
from keras.models import Sequential


logging.basicConfig(level=logging.INFO)


class ModelTrainer:

    def __init__(self, output_dir="output/models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scaler = StandardScaler()

    # ---------------------------------------------------------
    # REGRESSION MODELS (IMPROVED)
    # ---------------------------------------------------------

    def train_regression_models(self, X_train, X_test, y_train, y_test):

        # Scale features (important for Ridge/Lasso)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1)
        }

        results = {}

        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                results[name] = {
                    "model": model,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                }

                logging.info(f"{name} → MAE: {mae}, RMSE: {rmse}, R2: {r2}")

                # Save model
                joblib.dump(model, f"{self.output_dir}/{name}_model.pkl")

            except Exception as e:
                logging.error(f"{name} failed: {e}")

        # Select best model (based on RMSE)
        best_model_name = min(results, key=lambda x: results[x]["RMSE"])

        logging.info(f"Best Model: {best_model_name}")

        return results[best_model_name]["model"], results
    
        self.preds = model.predict(X_test)

        threshold = np.mean(self.y)

        self.y_binary = (self.y > threshold).astype(int)
        self.preds_binary = (self.preds > threshold).astype(int)

    # ---------------------------------------------------------
    # ARIMA MODEL (SAFE)
    # ---------------------------------------------------------

    def train_arima(self, series, order=(5, 1, 0)):

        try:
            series = series.dropna()

            model = ARIMA(series, order=order)
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=10)

            return model_fit, forecast

        except Exception as e:
            logging.error(f"ARIMA failed: {e}")
            return None, None

    # ---------------------------------------------------------
    # SARIMA MODEL (SAFE)
    # ---------------------------------------------------------

    def train_sarima(self, series, order=(1,1,1), seasonal_order=(1,1,1,12)):

        try:
            series = series.dropna()

            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order
            )

            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=10)

            return model_fit, forecast

        except Exception as e:
            logging.error(f"SARIMA failed: {e}")
            return None, None

    # ---------------------------------------------------------
    # PROPHET MODEL (FIXED)
    # ---------------------------------------------------------

    def train_prophet(self, df):

        try:
            if "date" not in df.columns:
                raise ValueError("Missing 'date' column for Prophet")

            prophet_df = df.rename(columns={
                "date": "ds",
                "demand_quantity": "y"
            })[["ds", "y"]]

            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=10)
            forecast = model.predict(future)

            return model, forecast

        except Exception as e:
            logging.error(f"Prophet failed: {e}")
            return None, None

    # ---------------------------------------------------------
    # GROUP FORECASTING (IMPROVED)
    # ---------------------------------------------------------

    def forecast_by_group(self, df, group_col):

        results = {}

        for group in df[group_col].dropna().unique():

            subset = df[df[group_col] == group]

            if len(subset) < 20:
                continue

            try:
                series = subset["demand_quantity"].dropna()

                model, forecast = self.train_arima(series)

                if forecast is not None:
                    results[group] = forecast

            except Exception as e:
                logging.warning(f"{group} failed: {e}")

        return results
    
    
    def train_advanced_models(self, X_train, X_test, y_train, y_test):

     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
     import numpy as np

     results = {}

    # -----------------------------
    # RANDOM FOREST
    # -----------------------------
     try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)

        results["RandomForest"] = {
            "model": rf,
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        }

     except Exception as e:
        logging.error(f"RF failed: {e}")

    # -----------------------------
    # XGBOOST
    # -----------------------------
     try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6
        )

        xgb_model.fit(X_train, y_train)

        preds = xgb_model.predict(X_test)

        results["XGBoost"] = {
            "model": xgb_model,
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        }

     except Exception as e:
        logging.error(f"XGBoost failed: {e}")

    # -----------------------------
    # NEURAL NETWORK (Feedforward)
    # -----------------------------
     try:
        model_nn = Sequential([
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1)
        ])

        model_nn.compile(
            optimizer="adam",
            loss="mse"
        )

        model_nn.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            verbose=0
        )

        preds = model_nn.predict(X_test).flatten()

        results["NeuralNet"] = {
            "model": model_nn,
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds)
        }

     except Exception as e:
        logging.error(f"NN failed: {e}")

     return results
 
    mlflow.start_run()

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()