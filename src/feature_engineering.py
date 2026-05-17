import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def create_features(df):

    df = df.copy()

    # -----------------------------
    # 1. TIME FEATURES (IMPORTANT)
    # -----------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week
        df["day_of_week"] = df["date"].dt.dayofweek

    # -----------------------------
    # 2. LAG FEATURES
    # -----------------------------
    if "demand_quantity" in df.columns:

        df["lag_1"] = df["demand_quantity"].shift(1)
        df["lag_7"] = df["demand_quantity"].shift(7)
        df["lag_14"] = df["demand_quantity"].shift(14)

    # -----------------------------
    # 3. ROLLING FEATURES
    # -----------------------------
    if "demand_quantity" in df.columns:

        df["rolling_mean_7"] = df["demand_quantity"].rolling(7).mean()
        df["rolling_std_7"] = df["demand_quantity"].rolling(7).std()

        df["rolling_mean_14"] = df["demand_quantity"].rolling(14).mean()

    # -----------------------------
    # 4. INTERACTION FEATURES
    # -----------------------------
    if "order_quantity" in df.columns and "inventory_level" in df.columns:
        df["inventory_pressure"] = df["order_quantity"] / (df["inventory_level"] + 1)

    if "lead_time" in df.columns and "order_quantity" in df.columns:
        df["lead_time_demand"] = df["lead_time"] * df["order_quantity"]

    # -----------------------------
    # 5. DOMAIN-SPECIFIC FEATURES
    # -----------------------------
    if "lead_time" in df.columns:
        df["lead_time_variability"] = df["lead_time"].rolling(7).std()

    if "supplier_rating" in df.columns:
        df["supplier_risk"] = 5 - df["supplier_rating"]

    # -----------------------------
    # 6. SEASONAL FEATURES
    # -----------------------------
    if "month" in df.columns:
        df["season_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["season_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # -----------------------------
    # 7. DEMAND VOLATILITY
    # -----------------------------
    if "demand_quantity" in df.columns:
        df["demand_volatility"] = df["demand_quantity"].rolling(7).std()

    # -----------------------------
    # 8. FILL NA (IMPORTANT after lag/rolling)
    # -----------------------------
    df = df.fillna(method="bfill").fillna(method="ffill")

    return df


# -----------------------------
# OPTIONAL: PCA (Dimensionality Reduction)
# -----------------------------

def apply_pca(df, n_components=5):

    numeric_df = df.select_dtypes(include=[np.number])

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(numeric_df)

    pca_df = pd.DataFrame(
        reduced,
        columns=[f"PC{i}" for i in range(n_components)]
    )

    return pca_df