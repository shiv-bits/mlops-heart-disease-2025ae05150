
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "heart_disease_pipeline.joblib"


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: int = Field(gt=0, le=300)
    chol: int = Field(gt=0, le=1000)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int = Field(gt=0, le=300)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=-10, le=15)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run training first."
        )
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational machine-learning demonstration. "
        "This API is not intended for medical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    model = app.state.model
    row = pd.DataFrame([patient.model_dump()])

    try:
        probabilities = model.predict_proba(row)[0]
        prediction = int(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    probability_presence = float(probabilities[1])
    confidence = float(probabilities[prediction])

    return {
        "prediction": prediction,
        "label": (
            "Heart disease present"
            if prediction == 1
            else "Heart disease absent"
        ),
        "confidence": round(confidence, 4),
        "probability_presence": round(probability_presence, 4),
        "disclaimer": "For educational use only, not medical diagnosis.",
    }
