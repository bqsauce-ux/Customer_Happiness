from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# load your trained model
model = joblib.load("models/best_model.joblib")


@app.get("/")
def home():
    return {"message": "Customer Happiness API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    # expected input: {"X1": 3, "X2": 4, ...}

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }
