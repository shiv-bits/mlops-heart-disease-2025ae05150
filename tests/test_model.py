
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "heart_disease_pipeline.joblib"

SAMPLE = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 1,
    "ca": 0,
    "thal": 6,
}


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_predicts_binary_class():
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(pd.DataFrame([SAMPLE]))
    assert int(prediction[0]) in (0, 1)


def test_model_probabilities_sum_to_one():
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(pd.DataFrame([SAMPLE]))[0]
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
