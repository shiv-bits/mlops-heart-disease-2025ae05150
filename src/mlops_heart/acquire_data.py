
import json
from pathlib import Path

from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # UCI Heart Disease dataset, dataset ID 45.
    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features
    targets = dataset.data.targets

    # The original target has values 0 through 4.
    df = features.copy()
    target_column = targets.columns[0]
    df[target_column] = targets[target_column].values

    output_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(output_path, index=False)

    metadata = {
        "source": "UCI Machine Learning Repository",
        "dataset_id": 45,
        "name": "Heart Disease",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "target_values": sorted(
            int(value) for value in df[target_column].dropna().unique()
        ),
    }

    (RAW_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved dataset: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Missing values:\n{df.isna().sum()}")


if __name__ == "__main__":
    main()
