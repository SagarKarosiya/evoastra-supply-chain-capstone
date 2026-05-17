import numpy as np
import optuna
import logging

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    KFold
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# Sagar Karosiya - BeginPlay
logging.basicConfig(level=logging.INFO)


class HyperparameterTuner:

    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    # ---------------------------------------------------------
    # K-FOLD CROSS VALIDATION
    # ---------------------------------------------------------
    def kfold_cv(self, model, folds=5):

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        scores = []

        for train_idx, val_idx in kf.split(self.X):
            X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

        return np.mean(scores)

    # ---------------------------------------------------------
    # TIME SERIES CV
    # ---------------------------------------------------------
    def time_series_cv(self, model, splits=5):

        tscv = TimeSeriesSplit(n_splits=splits)

        scores = []

        for train_idx, val_idx in tscv.split(self.X):
            X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

        return np.mean(scores)

    # ---------------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------------
    def grid_search_rf(self):

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }

        model = RandomForestRegressor()

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        grid.fit(self.X, self.y)

        logging.info(f"Best RF Params: {grid.best_params_}")

        return grid.best_estimator_

    # ---------------------------------------------------------
    # RANDOM SEARCH (XGBOOST)
    # ---------------------------------------------------------
    def random_search_xgb(self):

        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1]
        }

        model = xgb.XGBRegressor()

        rand = RandomizedSearchCV(
            model,
            param_dist,
            n_iter=10,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1
        )

        rand.fit(self.X, self.y)

        logging.info(f"Best XGB Params: {rand.best_params_}")

        return rand.best_estimator_

    # ---------------------------------------------------------
    # OPTUNA (BEST PRACTICE)
    # ---------------------------------------------------------
    def optuna_xgb(self, trials=20):

        def objective(trial):

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)
            }

            model = xgb.XGBRegressor(**params)

            score = self.kfold_cv(model)

            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)

        logging.info(f"Best Optuna Params: {study.best_params}")

        return xgb.XGBRegressor(**study.best_params)