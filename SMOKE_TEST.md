# Smoke test

After installing requirements:

```bash
python -m pytest -q
streamlit run app.py
```

The expected dashboard starts with the synthetic dataset and shows Overview, AI Risk,
Reconciliation, Incident Copilot and Webhook/API tabs.

For the API:

```bash
uvicorn api:app --port 8000
```

Then open `/health`.
