
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_disease_eda.csv"


def test_raw_dataset_exists():
    assert RAW_PATH.exists()


def test_dataset_shape_and_schema():
    df = pd.read_csv(RAW_PATH)
    assert df.shape == (303, 14)
    assert "num" in df.columns


def test_original_target_values():
    df = pd.read_csv(RAW_PATH)
    assert set(df["num"].unique()).issubset({0, 1, 2, 3, 4})


def test_expected_missing_values():
    df = pd.read_csv(RAW_PATH)
    assert df["ca"].isna().sum() == 4
    assert df["thal"].isna().sum() == 2


def test_processed_dataset_has_binary_target():
    df = pd.read_csv(PROCESSED_PATH)
    assert df.shape == (303, 15)
    assert set(df["target"].unique()) == {0, 1}
