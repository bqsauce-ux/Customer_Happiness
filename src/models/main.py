from fastapi import FastAPI
import joblib
import pandas as pd


model = joblib.load("best_model.joblib")
app = FastAPI()

# load your trained model


@app.get("/")
def home():
    return {"message": "Customer Happiness API is running "}


@app.post("/predict")
def predict(data: dict):
    # expected input: {"X1": 3, "X2": 4, ...}

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }
