
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "num"
sns.set_theme(style="whitegrid", context="notebook")


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # Validate the expected UCI dataset schema.
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} not found")
    if len(df) != 303:
        raise ValueError(f"Expected 303 rows, received {len(df)}")

    # Preserve original target and create a binary classification target.
    df["target"] = (df[TARGET] > 0).astype(int)

    # Reports before imputation, so original missingness remains visible.
    df.isna().sum().rename("missing_count").to_csv(
        REPORT_DIR / "missing_values.csv"
    )
    df.describe(include="all").transpose().to_csv(
        REPORT_DIR / "descriptive_statistics.csv"
    )

    # Save a cleaned copy for reproducibility. Missing feature values remain
    # present here; model pipelines perform imputation without data leakage.
    df.to_csv(PROCESSED_DIR / "heart_disease_eda.csv", index=False)

    # 1. Binary target balance.
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Target Balance")
    plt.xlabel("Target (0 = absent, 1 = present)")
    plt.ylabel("Patient count")
    save_plot("class_balance.png")

    # 2. Distributions of numeric measurements.
    numeric_cols = [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ]
    df[numeric_cols].hist(figsize=(12, 8), bins=18, edgecolor="black")
    plt.suptitle("Distributions of Numeric Features")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "feature_histograms.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Correlation heatmap of numeric columns.
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        cmap="coolwarm",
        center=0,
        annot=False,
    )
    plt.title("Feature Correlation Heatmap")
    save_plot("correlation_heatmap.png")

    # 4. Missingness visualization.
    plt.figure(figsize=(9, 4))
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        sns.barplot(x=missing.index, y=missing.values)
        plt.ylabel("Missing values")
        plt.title("Missing Values by Feature")
    else:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    save_plot("missing_values.png")

    # 5. Categorical feature distributions.
    categorical_cols = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for column, ax in zip(categorical_cols, axes.flat):
        sns.countplot(data=df, x=column, ax=ax)
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Categorical Feature Distributions")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "categorical_distributions.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Raw data shape: {df.shape}")
    print(f"Binary target counts:\n{df['target'].value_counts().sort_index()}")
    print(f"Saved cleaned dataset: {PROCESSED_DIR / 'heart_disease_eda.csv'}")
    print(f"Saved EDA reports and figures in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
