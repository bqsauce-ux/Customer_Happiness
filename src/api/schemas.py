from pydantic import BaseModel, Field
from typing import List

class CustomerHappinessRequest(BaseModel):
    X1: float = Field(
        ...,
        description="Customer rating for whether the order was delivered on time"
    )

    X2: float = Field(
        ...,
        description="Customer rating for whether the order contents matched expectations"
    )

    X3: float = Field(
        ...,
        description="Customer rating for whether everything desired was successfully ordered"
    )

    X4: float = Field(
        ...,
        description="Customer rating for whether the order was priced fairly"
    )

    X5: float = Field(
        ...,
        description="Customer rating of satisfaction with the courier service"
    )

    X6: float = Field(
        ...,
        description="Customer rating for how easy the app makes ordering"
    )
  
class PredictionResponse(BaseModel):
    predicted_happiness: float
    
