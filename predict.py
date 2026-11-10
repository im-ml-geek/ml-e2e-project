from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model and cutoff once when the API starts.
model_path = Path(__file__).resolve().with_name("lead_conversion_model.joblib")
artifact = joblib.load(model_path)
model = artifact["model"]
cutoff = artifact["cutoff"]

app = FastAPI(title="Lead Conversion API")


# Define the seven inputs the API expects for one lead.
class Lead(BaseModel):
    annual_income: float | None
    number_of_courses_viewed: int
    interaction_count: int
    lead_source: str | None
    industry: str | None
    employment_status: str | None
    location: str | None


@app.post("/predict")
def predict(lead: Lead):
    # Turn the request into one row and mark nulls as missing values.
    row = {
        name: float("nan") if value is None else value
        for name, value in lead.model_dump().items()
    }
    features = pd.DataFrame([row])

    # The saved pipeline prepares the inputs and scores class 1.
    probability = float(model.predict_proba(features)[0, 1])

    return {
        "conversion_probability": probability,
        "flag_for_follow_up": probability >= cutoff
    }