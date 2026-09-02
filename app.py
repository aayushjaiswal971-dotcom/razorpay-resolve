from __future__ import annotations

import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.data_engine import generate_demo_data, prepare_dataframe
from src.insights import answer_question, generate_insights, recommendation_for_row
from src.ml_engine import (
    add_anomaly_scores,
    add_failure_predictions,
    model_feature_importance,
    train_failure_model,
)
from src.reconciliation import reconcile, reconciliation_summary

load_dotenv()

st.set_page_config(
    page_title="RazorPay Resolve",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background: #0d0d0d; }
    .block-container { padding-top: 2rem; max-width: 1450px; }
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid #2b2b2b;
        border-radius: 18px;
        background: linear-gradient(135deg, #17130d 0%, #111111 70%);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; }
    .hero p { color: #bdb6aa; margin-top: .5rem; }
    .tag {
        display: inline-block; padding: .25rem .55rem; border-radius: 999px;
        background: #2a2115; color: #e8a63c; font-size: .78rem; font-weight: 700;
        letter-spacing: .05em; text-transform: uppercase;
    }
    .small { color: #a8a39a; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <span class="tag">Open Track · AI for Payment Operations</span>
      <h1>RazorPay Resolve</h1>
      <p>Predict payment failure risk, surface anomalies, reconcile settlements, and turn payment events into an actionable operations queue.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data")
    use_demo = st.toggle("Use demo dataset", value=True)
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    tolerance = st.number_input(
        "Settlement tolerance (₹)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.5,
    )
    st.caption(
        "Demo data is synthetic. For production, connect this pipeline to "
        "Razorpay payment/settlement webhooks or internal event streams."
    )

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        df = prepare_dataframe(raw)
        source_name = uploaded.name
    elif use_demo:
        df = generate_demo_data()
        source_name = "synthetic demo"
    else:
        st.info("Enable the demo dataset or upload a CSV to continue.")
        st.stop()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

# ML enrichment
try:
    bundle = train_failure_model(df)
    df = add_failure_predictions(df, bundle)
except ValueError as exc:
    st.warning(f"Failure model unavailable: {exc}")
    bundle = None
    df["failure_risk"] = 0.0
    df["risk_pct"] = 0.0
    df["risk_band"] = "Unknown"

df = add_anomaly_scores(df)
df = reconcile(df, tolerance=tolerance)

metrics = {
    "Transactions": f"{len(df):,}",
    "Volume": f"₹{df['amount'].sum():,.0f}",
    "Success rate": f"{(df['status'].eq('success').mean() * 100):.1f}%",
    "High-risk": f"{(df['failure_risk'].ge(0.65).sum()):,}",
    "Anomalies": f"{df['anomaly_flag'].sum():,}",
    "Recon mismatches": f"{df['recon_status'].eq('Mismatch').sum():,}",
}

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, value)

tabs = st.tabs(
    ["Overview", "AI Risk", "Reconciliation", "Incident Copilot", "Webhook/API"]
)

with tabs[0]:
    left, right = st.columns(2)
    with left:
        trend = (
            df.set_index("timestamp")
            .resample("D")
            .agg(
                success_rate=("failure_flag", lambda s: 100 * (1 - s.mean())),
                volume=("amount", "sum"),
            )
            .reset_index()
        )
        fig = px.line(
            trend,
            x="timestamp",
            y="success_rate",
            markers=True,
            title="Daily payment success rate",
        )
        fig.update_yaxes(range=[0, 100], title="Success rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        method = (
            df.groupby("method", as_index=False)["failure_flag"]
            .mean()
            .assign(failure_rate=lambda x: x["failure_flag"] * 100)
        )
        fig = px.bar(
            method,
            x="method",
            y="failure_rate",
            title="Failure rate by payment method",
            text_auto=".1f",
        )
        fig.update_yaxes(title="Failure rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI-generated operational signals")
    for insight in generate_insights(df):
        st.write("• " + insight)

    st.subheader("Highest-priority transactions")
    priority = df.sort_values(
        ["risk_pct", "abs_settlement_gap"], ascending=False
    ).head(15).copy()
    priority["recommended_action"] = priority.apply(recommendation_for_row, axis=1)
    st.dataframe(
        priority[
            [
                "transaction_id",
                "amount",
                "status",
                "method",
                "risk_pct",
                "risk_band",
                "anomaly_flag",
                "recon_status",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Payment failure prediction")
    st.caption(
        "The baseline model is an explainable logistic-regression classifier trained on the current dataset. "
        "It is a decision-support layer, not a fraud verdict."
    )
    if bundle is not None:
        c1, c2 = st.columns(2)
        c1.metric("Training AUC", f"{bundle.auc:.3f}" if bundle.auc is not None else "N/A")
        c2.metric("Training accuracy", f"{bundle.accuracy:.1%}" if bundle.accuracy is not None else "N/A")

        importance = model_feature_importance(bundle)
        fig = px.bar(
            importance.sort_values("abs_impact"),
            x="abs_impact",
            y="feature",
            orientation="h",
            title="Most influential model signals",
        )
        st.plotly_chart(fig, use_container_width=True)

    risk_view = df[
        [
            "transaction_id", "timestamp", "amount", "method", "gateway_status",
            "retry_count", "latency_ms", "device_risk", "merchant_risk",
            "risk_pct", "risk_band", "anomaly_flag",
        ]
    ].sort_values("risk_pct", ascending=False)

    st.download_button(
        "Download scored transactions",
        data=risk_view.to_csv(index=False).encode("utf-8"),
        file_name="razorpay_resolve_scored_transactions.csv",
        mime="text/csv",
    )
    st.dataframe(risk_view.head(100), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Settlement reconciliation queue")
    recon = reconciliation_summary(df, tolerance=tolerance)
    c1, c2, c3 = st.columns(3)
    c1.metric("Checked", f"{recon['checked']:,}")
    c2.metric("Matched", f"{recon['matched']:,}")
    c3.metric("Mismatches", f"{recon['mismatches']:,}")

    recon_view = df[
        [
            "transaction_id", "timestamp", "amount", "status",
            "expected_settlement", "settlement_amount",
            "recon_difference", "recon_status", "recon_priority",
        ]
    ].sort_values("recon_difference", key=lambda s: s.abs(), ascending=False)

    st.dataframe(recon_view.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download reconciliation queue",
        data=recon_view.to_csv(index=False).encode("utf-8"),
        file_name="razorpay_resolve_reconciliation.csv",
        mime="text/csv",
    )

with tabs[3]:
    st.subheader("Incident Copilot")
    st.caption(
        "Ask operational questions about the current dataset. The demo uses deterministic analytics so it works without an external LLM or API key."
    )
    question = st.text_input(
        "Ask Resolve",
        placeholder="Why did failures increase? / Which settlements need review? / What should we do next?",
    )
    if question:
        st.info(answer_question(question, df))

    st.markdown("**Quick prompts**")
    quick = st.columns(4)
    prompts = [
        "Why are payments failing?",
        "Which settlements need review?",
        "Show me risk signals.",
        "What should we do next?",
    ]
    for col, prompt in zip(quick, prompts):
        if col.button(prompt, use_container_width=True):
            st.info(answer_question(prompt, df))

    st.subheader("Routing sandbox")
    selected_id = st.selectbox(
        "Select a transaction",
        df["transaction_id"].head(300).tolist(),
    )
    row = df.loc[df["transaction_id"] == selected_id].iloc[0]
    st.write(
        f"**{selected_id}** · ₹{row['amount']:,.2f} · "
        f"{row['method'].upper()} · risk **{row['risk_pct']:.1f}%**"
    )
    st.success(recommendation_for_row(row))

with tabs[4]:
    st.subheader("Production integration path")
    st.markdown(
        """
        **Webhook receiver:** `api.py` accepts Razorpay-style event payloads,
        verifies `X-Razorpay-Signature` when a secret is configured, de-duplicates
        events by event ID, and writes them to a durable JSONL queue for downstream processing.

        **Run it locally**
        ```bash
        uvicorn api:app --reload --port 8000
        ```

        **Health check**
        ```text
        GET http://localhost:8000/health
        ```

        **Webhook**
        ```text
        POST http://localhost:8000/webhook/razorpay
        ```

        In production, replace the JSONL store with Kafka/SQS/PubSub + a database,
        process heavy analytics asynchronously, and keep the webhook response fast.
        """
    )

    event_file = os.getenv("WEBHOOK_EVENT_FILE", "data/webhook_events.jsonl")
    if os.path.exists(event_file):
        st.caption(f"Local webhook event store: {event_file}")
        with open(event_file, "rb") as handle:
            st.download_button(
                "Download received webhook events",
                data=handle.read(),
                file_name="webhook_events.jsonl",
                mime="application/jsonl",
            )

st.divider()
st.markdown(
    '<div class="small">RazorPay Resolve is a hackathon prototype. It uses synthetic/demo data and should not be used for production payment, fraud, settlement, or merchant decisions without security, compliance, model validation, and human-oversight controls.</div>',
    unsafe_allow_html=True,
)
