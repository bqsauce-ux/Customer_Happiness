import joblib
import pandas as pd
from datetime import datetime
from schemas import CustomerHappinessRequest, PredictionResponse
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "src/models/best_model.joblib"
model = joblib.load(MODEL_PATH)

# load your trained model

def predict_happiness(request: CustomerHappinessRequest) -> PredictionResponse:
    
    input_data = pd.DataFrame([request.dict()])
    input_data = input_data[['X1', 'X5', 'X6']]    
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    prediction = model.predict(input_data)

    return PredictionResponse(
        predicted_happiness= prediction
    )
