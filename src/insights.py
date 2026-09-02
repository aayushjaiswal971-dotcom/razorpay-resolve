from __future__ import annotations

import pandas as pd

from .data_engine import summary_metrics


def recommendation_for_row(row: pd.Series) -> str:
    risk = float(row.get("failure_risk", 0))
    latency = float(row.get("latency_ms", 0))
    retries = int(row.get("retry_count", 0))
    gateway = str(row.get("gateway_status", ""))

    if gateway == "timeout" or latency > 2200:
        return "Route away from degraded path; retry after a short backoff."
    if risk >= 0.65 and retries >= 2:
        return "Stop blind retries; show alternate payment method and trigger support."
    if risk >= 0.45:
        return "Offer an alternate method and one controlled retry."
    return "Allow normal flow; monitor only."


def generate_insights(df: pd.DataFrame) -> list[str]:
    m = summary_metrics(df)
    insights: list[str] = []

    if m["success_rate"] < 90:
        insights.append(
            f"Payment success is {m['success_rate']:.1f}%. Prioritise gateway/method cohorts with the highest failure concentration."
        )
    else:
        insights.append(
            f"Payment success is healthy at {m['success_rate']:.1f}%; focus on long-tail failures and operational exceptions."
        )

    method_fail = (
        df.groupby("method")["failure_flag"].mean().sort_values(ascending=False)
    )
    if not method_fail.empty:
        top_method = method_fail.index[0]
        insights.append(
            f"{top_method.upper()} has the highest failure rate in this dataset ({method_fail.iloc[0] * 100:.1f}%)."
        )

    if m["avg_latency_ms"] > 1100:
        insights.append(
            f"Average latency is {m['avg_latency_ms']:.0f} ms. Investigate slow gateway cohorts before adding more retries."
        )

    if m["settlement_mismatch_count"] > 0:
        insights.append(
            f"{m['settlement_mismatch_count']} settlement records exceed the reconciliation tolerance; send those cases to ops review."
        )

    anomaly_count = int(df.get("anomaly_flag", pd.Series(dtype=int)).sum())
    if anomaly_count:
        insights.append(
            f"{anomaly_count} transactions look operationally unusual; inspect amount, latency, retries and settlement gaps together."
        )

    return insights


def answer_question(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    m = summary_metrics(df)

    if any(k in q for k in ["why", "failure", "fail"]):
        method = (
            df.groupby("method")["failure_flag"].mean()
            .sort_values(ascending=False)
            .head(3)
        )
        gateway = (
            df.groupby("gateway_status")["failure_flag"].mean()
            .sort_values(ascending=False)
            .head(3)
        )
        method_text = ", ".join(
            f"{idx.upper()} {value * 100:.1f}%" for idx, value in method.items()
        )
        gateway_text = ", ".join(
            f"{idx} {value * 100:.1f}%" for idx, value in gateway.items()
        )
        return (
            f"The main signal is a {m['success_rate']:.1f}% success rate. "
            f"Failure is concentrated in methods ({method_text}) and gateway states ({gateway_text}). "
            f"Average latency is {m['avg_latency_ms']:.0f} ms. "
            "The recommended response is cohort-level routing/retry control rather than retrying every failed payment."
        )

    if any(k in q for k in ["settlement", "recon", "reconcile"]):
        return (
            f"There are {m['settlement_mismatch_count']} settlement exceptions with a combined absolute gap "
            f"of ₹{m['settlement_gap_value']:,.2f}. Prioritise the largest gaps first and reconcile against "
            "the settlement reference/UTR before any merchant adjustment."
        )

    if any(k in q for k in ["fraud", "risk", "suspicious", "anomaly"]):
        anomaly_count = int(df.get("anomaly_flag", pd.Series(dtype=int)).sum())
        high_risk = int((df.get("failure_risk", pd.Series(dtype=float)) >= 0.65).sum())
        return (
            f"The dashboard flags {anomaly_count} operational anomalies and {high_risk} high-risk transactions. "
            "Use the risk score as a prioritisation signal, not an automatic fraud verdict; combine it with "
            "Razorpay's existing risk controls and human review for high-impact decisions."
        )

    if any(k in q for k in ["improve", "recommend", "next", "action"]):
        return (
            "Next best actions: (1) route around degraded gateway cohorts, "
            "(2) cap repeated retries, (3) surface alternate payment methods, "
            "(4) auto-queue settlement mismatches, and (5) monitor webhook delivery/idempotency."
        )

    return (
        f"I can explain failures, settlement mismatches, anomalies or recommended actions. "
        f"Current dataset: {m['transactions']} transactions, {m['success_rate']:.1f}% success, "
        f"₹{m['volume']:,.0f} attempted volume."
    )
