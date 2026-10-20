
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "heart_disease_pipeline.joblib"

HTTP_REQUESTS = Counter(
    "heart_disease_http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "heart_disease_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

PREDICTIONS = Counter(
    "heart_disease_predictions_total",
    "Total successful prediction requests",
    ["prediction"],
)


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
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    # Exclude metrics scraping to prevent monitoring traffic from
    # inflating the application's own request metrics.
    if request.url.path.rstrip("/") == "/metrics":
        return await call_next(request)

    method = request.method
    endpoint = request.url.path
    started = perf_counter()
    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        HTTP_REQUESTS.labels(method, endpoint, status).inc()
        HTTP_REQUEST_DURATION.labels(method, endpoint).observe(
            perf_counter() - started
        )


app.mount("/metrics", make_asgi_app())


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

    PREDICTIONS.labels(str(prediction)).inc()

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
