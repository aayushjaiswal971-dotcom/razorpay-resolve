from src.data_engine import generate_demo_data
from src.ml_engine import add_anomaly_scores, add_failure_predictions, train_failure_model
from src.reconciliation import reconcile


def test_demo_data_shape():
    df = generate_demo_data(250)
    assert len(df) == 250
    assert {"success", "failed"} >= set(df["status"].unique())


def test_model_scores():
    df = generate_demo_data(500)
    bundle = train_failure_model(df)
    scored = add_failure_predictions(df, bundle)
    assert scored["failure_risk"].between(0, 1).all()


def test_anomaly_and_reconciliation():
    df = generate_demo_data(300)
    scored = add_anomaly_scores(df)
    assert "anomaly_flag" in scored.columns
    recon = reconcile(scored)
    assert "recon_status" in recon.columns
