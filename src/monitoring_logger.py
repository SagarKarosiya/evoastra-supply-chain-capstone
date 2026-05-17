import pandas as pd
import os
from datetime import datetime

LOG_FILE = "monitoring/metrics_log.csv"

os.makedirs("monitoring", exist_ok=True)


def log_metrics(rmse, drift_score):

    data = {
        "timestamp": datetime.now(),
        "rmse": rmse,
        "drift_score": drift_score
    }

    df = pd.DataFrame([data])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)