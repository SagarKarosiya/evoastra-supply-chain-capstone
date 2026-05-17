from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging



from src.preprocessing import preprocess_data
from src.feature_engineering import create_features

app = FastAPI()

model = joblib.load("output/models/Linear_model.pkl")  # or best_model.pkl


class InputData(BaseModel):
    supplier_rating: float
    delivery_delay: float
    demand_quantity: float
    inventory_level: float


@app.get("/")
def home():
    return {"status": "API Running 🚀"}


@app.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([data.dict()])

    df = preprocess_data(df)
    df = create_features(df)

    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}

logging.info(f"Prediction: {prediction}")
logging.info(f"Input: {data}")