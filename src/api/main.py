from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_happiness
from schemas import CustomerHappinessRequest, PredictionResponse

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Customer Happiness Prediction API",
    description=(
        "An API for predicting customer happiness using different features."
        "This application is part of the Apziva's AI Residency Program. "
        "Authored by Melanie Qu."
    ),
    version="1.0.0",
    contact={
        "name": "Melanie Qu",
        "email": "bqsauce@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CustomerHappinessRequest):
    return predict_happiness(request)
