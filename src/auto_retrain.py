import pandas as pd

df = pd.read_csv("monitoring/metrics_log.csv")

latest = df.iloc[-1]

if latest["rmse"] > 40 or latest["drift_score"] > 0.3:
    print("🔄 Triggering retraining...")
    import os
    os.system("python run_pipeline.py")