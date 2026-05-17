import time
import numpy as np
from src.monitoring_logger import log_metrics
from src.monitoring import DriftMonitor

print("🚀 Starting Monitoring Simulation...")

while True:

    # Simulated values
    rmse = np.random.uniform(10, 50)
    drift_score = np.random.uniform(0, 1)

    log_metrics(rmse, drift_score)

    print(f"Logged RMSE: {rmse:.2f}, Drift: {drift_score:.2f}")

    time.sleep(5)  # every 5 seconds
    
    # simulate reference vs current
    drift_score = np.random.uniform(0, 1)

    if drift_score > 0.3:
     print("⚠️ Drift detected!")