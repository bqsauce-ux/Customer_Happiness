import joblib
import pandas as pd
from datetime import datetime
from schemas import CustomerHappinessRequest, PredictionResponse

MODEL_PATH = "src/models/best_model.joblib"
model = joblib.load(MODEL_PATH)

# load your trained model

def predict_happiness(request: CustomerHappinessRequest) -> PredictionResponse:
    
    input_data = pd.DataFrame([request.dict()])
    prediction = model.predict(input_data)[0]

    return PredictionResponse(
        predicted_happiness= prediction
    )
