from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "amount",
    "status",
    "method",
    "retry_count",
    "latency_ms",
    "device_risk",
    "merchant_risk",
    "gateway_status",
    "settlement_amount",
    "expected_settlement",
]

COLUMN_ALIASES = {
    "id": "transaction_id",
    "payment_id": "transaction_id",
    "created_at": "timestamp",
    "date": "timestamp",
    "value": "amount",
    "payment_amount": "amount",
    "payment_method": "method",
    "payment_status": "status",
    "retries": "retry_count",
    "latency": "latency_ms",
    "device_score": "device_risk",
    "merchant_score": "merchant_risk",
    "gateway": "gateway_status",
    "settled_amount": "settlement_amount",
    "expected_amount": "expected_settlement",
}


def _clean_status(value: object) -> str:
    value = str(value).strip().lower()
    if value in {"captured", "success", "successful", "paid", "completed", "authorized"}:
        return "success"
    if value in {"failed", "failure", "declined", "error"}:
        return "failed"
    return value


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    rename = {c: COLUMN_ALIASES[c] for c in out.columns if c in COLUMN_ALIASES}
    return out.rename(columns=rename)


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(df)
    missing = validate_columns(out)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed.")

    numeric_cols = [
        "amount", "retry_count", "latency_ms", "device_risk",
        "merchant_risk", "settlement_amount", "expected_settlement"
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[numeric_cols].isna().any().any():
        bad = out[numeric_cols].columns[
            out[numeric_cols].isna().any()
        ].tolist()
        raise ValueError("Numeric columns contain invalid values: " + ", ".join(bad))

    out["status"] = out["status"].map(_clean_status)
    out["method"] = out["method"].astype(str).str.lower().str.strip()
    out["gateway_status"] = out["gateway_status"].astype(str).str.lower().str.strip()
    out["transaction_id"] = out["transaction_id"].astype(str)

    out["failure_flag"] = (out["status"] == "failed").astype(int)
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["amount_log"] = np.log1p(out["amount"].clip(lower=0))
    out["settlement_gap"] = out["expected_settlement"] - out["settlement_amount"]
    out["abs_settlement_gap"] = out["settlement_gap"].abs()

    return out


def generate_demo_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic, realistic-looking payment operations data.

    This is synthetic data for the hackathon demo; it is not Razorpay production data.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-07-01")
    timestamps = start + pd.to_timedelta(
        rng.integers(0, 31 * 24 * 60, size=n), unit="m"
    )

    methods = rng.choice(
        ["upi", "card", "netbanking", "wallet"],
        size=n,
        p=[0.52, 0.28, 0.15, 0.05],
    )
    gateways = rng.choice(
        ["healthy", "healthy", "healthy", "degraded", "timeout"],
        size=n,
        p=[0.50, 0.25, 0.12, 0.10, 0.03],
    )

    amount = np.round(np.exp(rng.normal(7.1, 1.0, n)), 2).clip(25, 150000)
    retry_count = rng.choice([0, 1, 2, 3], size=n, p=[0.78, 0.15, 0.05, 0.02])
    latency = np.round(
        rng.normal(750, 230, n)
        + np.where(gateways == "degraded", 650, 0)
        + np.where(gateways == "timeout", 1600, 0),
        0,
    ).clip(80, 5000)
    device_risk = np.round(rng.beta(2.2, 7.0, n), 3)
    merchant_risk = np.round(rng.beta(2.0, 8.0, n), 3)

    hour = timestamps.hour.to_numpy()
    peak_penalty = np.where((hour >= 19) & (hour <= 23), 0.045, 0.0)
    method_penalty = np.select(
        [methods == "netbanking", methods == "card"],
        [0.025, 0.015],
        default=0.0,
    )
    gateway_penalty = np.select(
        [gateways == "degraded", gateways == "timeout"],
        [0.10, 0.23],
        default=0.0,
    )
    risk_score = (
        0.025
        + 0.18 * device_risk
        + 0.12 * merchant_risk
        + 0.028 * retry_count
        + np.maximum(latency - 900, 0) / 12000
        + peak_penalty
        + method_penalty
        + gateway_penalty
    )
    failure_prob = np.clip(risk_score, 0.015, 0.72)
    failed = rng.random(n) < failure_prob
    status = np.where(failed, "failed", "success")

    fee = np.round(amount * 0.018 + 2.0, 2)
    expected_settlement = np.where(status == "success", amount - fee, 0.0)
    settlement_amount = expected_settlement.copy()

    # Inject a small number of reconciliation exceptions.
    eligible = np.where(status == "success")[0]
    mismatch_count = max(8, int(len(eligible) * 0.012))
    mismatch_idx = rng.choice(eligible, size=mismatch_count, replace=False)
    settlement_amount[mismatch_idx] -= np.round(
        rng.uniform(1, 85, size=mismatch_count), 2
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"pay_demo_{i:06d}" for i in range(n)],
            "timestamp": timestamps,
            "amount": amount,
            "status": status,
            "method": methods,
            "retry_count": retry_count,
            "latency_ms": latency,
            "device_risk": device_risk,
            "merchant_risk": merchant_risk,
            "gateway_status": gateways,
            "settlement_amount": settlement_amount,
            "expected_settlement": expected_settlement,
        }
    )
    return prepare_dataframe(df)


def summary_metrics(df: pd.DataFrame) -> dict[str, float]:
    total = len(df)
    success = int((df["status"] == "success").sum())
    failed = int((df["status"] == "failed").sum())
    return {
        "transactions": total,
        "volume": float(df["amount"].sum()),
        "success_rate": (success / total * 100) if total else 0.0,
        "failed": failed,
        "failed_value": float(df.loc[df["status"] == "failed", "amount"].sum()),
        "avg_latency_ms": float(df["latency_ms"].mean()) if total else 0.0,
        "settlement_mismatch_count": int((df["abs_settlement_gap"] > 1.0).sum()),
        "settlement_gap_value": float(df["abs_settlement_gap"].sum()),
    }
