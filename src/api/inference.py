import joblib
import pandas as pd
from datetime import datetime
from schemas import CustomerHappinessRequest, PredictionResponse

MODEL_PATH = "src/models/best_model.joblib"
model = joblib.load(MODEL_PATH)

# load your trained model

def predict_happiness(request: CustomerHappinessRequest) -> PredictionResponse:
    
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return PredictionResponse(
        predicted_happiness= prediction
    )
