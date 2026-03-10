import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

EXP_ID = "807843010825906201"
MODEL_ID = "m-642c2301f9f14021bdb1bc2492e42610"

model_path = f"/app/mlruns/{EXP_ID}/models/{MODEL_ID}/artifacts"

model = mlflow.xgboost.load_model(model_path)

class BuildingInputs(BaseModel):
    X1: float; X2: float; X3: float; X4: float
    X5: float; X6: float; X7: float; X8: float

@app.get("/")
def home():
    return {"message": "BEAM Energy Prediction API is running"}

@app.post("/predict")
def get_prediction(data: BuildingInputs):
    # Convert data to DataFrame
    features = data.model_dump()
    input_df = pd.DataFrame([features])
    
    prediction = model.predict(input_df)
    
    return {
        "Heating_Load": float(prediction[0][0]),
        "Cooling_Load": float(prediction[0][1])
    }