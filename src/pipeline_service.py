import joblib
import pandas as pd

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features


class PipelineService:

    def __init__(self, model_path="output/models/best_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, input_data: dict):

        df = pd.DataFrame([input_data])

        df = preprocess_data(df)
        df = create_features(df)

        prediction = self.model.predict(df)[0]

        return {
            "prediction": float(prediction)
        }