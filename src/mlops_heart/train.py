
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "heart_disease_eda.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"


def make_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": (
            make_pipeline(LogisticRegression(max_iter=2000)),
            {
                "classifier__C": [0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "RandomForest": (
            make_pipeline(RandomForestClassifier(random_state=42)),
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }

    tracking_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-classification")

    results = []
    fitted_models = {}
    roc_data = {}

    for model_name, (pipeline, param_grid) in models.items():
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )

        with mlflow.start_run(run_name=model_name) as run:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            predictions = best_model.predict(X_test)
            probabilities = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(
                    y_test, predictions, zero_division=0
                )),
                "recall": float(recall_score(
                    y_test, predictions, zero_division=0
                )),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "cv_roc_auc": float(search.best_score_),
            }

            mlflow.log_params({
                "model": model_name,
                "cv_folds": 5,
                "test_size": 0.2,
                "random_state": 42,
                **{k: str(v) for k, v in search.best_params_.items()},
            })
            mlflow.log_metrics(metrics)

            report = classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
            report_path = REPORT_DIR / f"{model_name}_classification_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            cv_path = REPORT_DIR / f"{model_name}_cv_results.csv"
            pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)

            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots(figsize=(5, 4))
            from sklearn.metrics import ConfusionMatrixDisplay
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Absent", "Present"],
            ).plot(ax=ax, colorbar=False)
            ax.set_title(f"{model_name} Confusion Matrix")
            fig.tight_layout()
            cm_path = FIGURE_DIR / f"{model_name.lower()}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=160)
            plt.close(fig)

            RocCurveDisplay.from_predictions(y_test, probabilities, name=model_name)
            roc_data[model_name] = (y_test.copy(), probabilities.copy())
            results.append({
                "model": model_name,
                **metrics,
                "best_params": json.dumps(search.best_params_, default=str),
                "run_id": run.info.run_id,
            })
            fitted_models[model_name] = best_model

            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )   

            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(cv_path))
            mlflow.log_artifact(str(cm_path))

            print(f"\n{model_name}")
            print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
            print(f"Test metrics: {metrics}")
            print(f"Best parameters: {search.best_params_}")

    comparison = pd.DataFrame(results).sort_values(
        "cv_roc_auc", ascending=False
    )
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_comparison.json").write_text(
        comparison.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    # Select the final model using cross-validation ROC-AUC, not test performance.
    selected_name = str(comparison.iloc[0]["model"])
    selected_model = fitted_models[selected_name]

    joblib.dump(selected_model, ARTIFACT_DIR / "heart_disease_pipeline.joblib")
    metadata = {
        "selected_model": selected_name,
        "selection_metric": "cv_roc_auc",
        "cv_roc_auc": float(comparison.iloc[0]["cv_roc_auc"]),
        "features": FEATURES,
        "target": TARGET,
        "test_metrics": {
            row["model"]: {
                key: float(row[key])
                for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            }
            for row in results
        },
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Final comparison ROC curve.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (actual, probs) in roc_data.items():
        RocCurveDisplay.from_predictions(actual, probs, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title("ROC Curves: Model Comparison")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)

    print("\nModel comparison:")
    print(comparison[["model", "cv_roc_auc", "accuracy", "f1", "roc_auc"]])
    print(f"\nSelected model: {selected_name}")
    print(f"Saved model: {ARTIFACT_DIR / 'heart_disease_pipeline.joblib'}")


if __name__ == "__main__":
    main()
