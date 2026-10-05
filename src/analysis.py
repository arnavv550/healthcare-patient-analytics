"""
Healthcare Patient Analytics & Readmission Risk

Downloads the UCI Diabetes 130-US Hospitals dataset, performs basic
cleaning, feature engineering, exploratory analysis, segmentation,
and a baseline interpretable classification model.

This is a portfolio project, not a clinical decision-support system.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# Project paths
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# -----------------------
# Load dataset
# -----------------------

print("Downloading UCI dataset...")
dataset = fetch_ucirepo(id=296)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# The UCI target is the readmitted column.
target_col = y.columns[0]

df = X.copy()
df[target_col] = y[target_col]

# Replace UCI missing-value markers.
df = df.replace("?", np.nan)


# -----------------------
# Target variable
# -----------------------

# Binary target:
# 1 = readmitted within 30 days
# 0 = otherwise
df["readmitted_30d"] = (df[target_col] == "<30").astype(int)


# -----------------------
# Feature engineering
# -----------------------

df["prior_utilization"] = (
    pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0)
    + pd.to_numeric(df["number_emergency"], errors="coerce").fillna(0)
    + pd.to_numeric(df["number_outpatient"], errors="coerce").fillna(0)
)


# -----------------------
# Analytical sample
# -----------------------

sample_cols = [
    "age",
    "gender",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "prior_utilization",
    "readmitted_30d",
]

sample_cols = [c for c in sample_cols if c in df.columns]

df[sample_cols].to_csv(
    OUT / "analytical_sample.csv",
    index=False
)


# -----------------------
# KPI summary
# -----------------------

kpis = {
    "total_encounters": len(df),
    "readmitted_within_30_days": int(
        df["readmitted_30d"].sum()
    ),
    "readmission_rate_pct": round(
        df["readmitted_30d"].mean() * 100,
        2
    ),
    "avg_length_of_stay": round(
        pd.to_numeric(
            df["time_in_hospital"],
            errors="coerce"
        ).mean(),
        2
    ),
    "avg_medications": round(
        pd.to_numeric(
            df["num_medications"],
            errors="coerce"
        ).mean(),
        2
    ),
}

pd.DataFrame([kpis]).to_csv(
    OUT / "kpi_summary.csv",
    index=False
)


# -----------------------
# Segmentation analysis
# -----------------------

if "age" in df.columns:

    age_summary = (
        df.groupby(
            "age",
            dropna=False
        )["readmitted_30d"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "encounters",
                "mean": "readmission_rate"
            }
        )
    )

    age_summary["readmission_rate_pct"] = (
        age_summary["readmission_rate"] * 100
    ).round(2)

    age_summary.drop(
        columns="readmission_rate"
    ).to_csv(
        OUT / "readmission_by_age.csv",
        index=False
    )


# -----------------------
# Visualization
# -----------------------

if "age" in df.columns:

    plot_df = (
        df.groupby("age")["readmitted_30d"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))

    plot_df.plot(kind="bar")

    plt.ylabel("Readmission rate (%)")
    plt.xlabel("Age group")
    plt.title(
        "30-Day Readmission Rate by Age Group"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "readmission_by_age.png",
        dpi=180
    )

    plt.close()


# -----------------------
# Baseline predictive model
# -----------------------

candidate_features = [
    "age",
    "gender",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "prior_utilization",
    "admission_type_id",
]

features = [
    c for c in candidate_features
    if c in df.columns
]

model_df = df[
    features + ["readmitted_30d"]
].copy()

X_model = model_df[features]
y_model = model_df["readmitted_30d"]


# Identify categorical and numerical variables
categorical = X_model.select_dtypes(
    include=["object"]
).columns.tolist()

numeric = [
    c for c in X_model.columns
    if c not in categorical
]


# Numerical preprocessing
numeric_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
])


# Categorical preprocessing
categorical_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    ),
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipe,
        numeric
    ),
    (
        "cat",
        categorical_pipe,
        categorical
    ),
])


# Logistic regression model
model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    ),
])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.20,
    random_state=42,
    stratify=y_model
)


# Train model
model.fit(
    X_train,
    y_train
)


# Predictions
prob = model.predict_proba(
    X_test
)[:, 1]

pred = (
    prob >= 0.5
).astype(int)


# Model evaluation
auc = roc_auc_score(
    y_test,
    prob
)


# Save model report
with open(
    OUT / "model_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"ROC-AUC: {auc:.4f}\n\n"
    )

    f.write(
        classification_report(
            y_test,
            pred
        )
    )


# -----------------------
# Final output
# -----------------------

print("\nProject completed.")

print(
    f"Rows analysed: {len(df):,}"
)

print(
    f"30-day readmission rate: "
    f"{kpis['readmission_rate_pct']}%"
)

print(
    f"Baseline ROC-AUC: {auc:.4f}"
)

print(
    f"Outputs saved to: {OUT}"
)
