import sys
import os
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.model_training import train_model
from src.evaluate import evaluate


sys.path.append(os.path.abspath(os.path.dirname(__file__)))


df = load_data("data/raw/msr.csv")
df = clean_data(df)
df = create_features(df)

model, X_test, y_test = train_model(df)
evaluate(model, X_test, y_test)