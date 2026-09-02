from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "amount",
    "retry_count",
    "latency_ms",
    "device_risk",
    "merchant_risk",
    "hour",
    "day_of_week",
    "is_weekend",
    "method",
    "gateway_status",
]
NUMERIC_FEATURES = [
    "amount",
    "retry_count",
    "latency_ms",
    "device_risk",
    "merchant_risk",
    "hour",
    "day_of_week",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["method", "gateway_status"]


@dataclass
class ModelBundle:
    pipeline: Pipeline
    auc: float | None
    accuracy: float | None


def train_failure_model(df: pd.DataFrame) -> ModelBundle:
    """Train an explainable baseline model on the supplied transaction data."""
    if df["failure_flag"].nunique() < 2:
        raise ValueError("Need both successful and failed transactions to train the model.")

    X = df[FEATURES]
    y = df["failure_flag"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X, y)

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs) if y.nunique() == 2 else None
    accuracy = accuracy_score(y, preds)

    return ModelBundle(pipeline=pipeline, auc=auc, accuracy=accuracy)


def add_failure_predictions(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = df.copy()
    out["failure_risk"] = bundle.pipeline.predict_proba(out[FEATURES])[:, 1]
    out["risk_pct"] = out["failure_risk"] * 100
    out["risk_band"] = pd.cut(
        out["failure_risk"],
        bins=[-0.01, 0.25, 0.55, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return out


def add_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    features = [
        "amount",
        "latency_ms",
        "retry_count",
        "device_risk",
        "merchant_risk",
        "abs_settlement_gap",
    ]
    model = IsolationForest(
        n_estimators=250,
        contamination=0.03,
        random_state=42,
    )
    labels = model.fit_predict(out[features])
    raw_scores = -model.score_samples(out[features])
    out["anomaly_flag"] = (labels == -1).astype(int)
    # Normalize only for display; it is not a probability.
    lo, hi = float(raw_scores.min()), float(raw_scores.max())
    out["anomaly_score"] = (
        (raw_scores - lo) / (hi - lo) if hi > lo else 0.0
    )
    return out


def model_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    pre = bundle.pipeline.named_steps["preprocessor"]
    model = bundle.pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    coefs = model.coef_[0]
    result = pd.DataFrame(
        {
            "feature": names,
            "impact": coefs,
            "abs_impact": np.abs(coefs),
        }
    ).sort_values("abs_impact", ascending=False)
    return result.head(12)
