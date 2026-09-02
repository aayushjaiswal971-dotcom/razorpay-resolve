from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

load_dotenv()

app = FastAPI(
    title="RazorPay Resolve Webhook Gateway",
    version="1.0.0",
)

EVENT_FILE = Path(os.getenv("WEBHOOK_EVENT_FILE", "data/webhook_events.jsonl"))
SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
ALLOW_UNSIGNED_DEMO = os.getenv("ALLOW_UNSIGNED_DEMO_WEBHOOKS", "true").lower() == "true"

_seen_event_ids: set[str] = set()


def verify_signature(raw_body: bytes, signature: str | None) -> bool:
    if not SECRET:
        return ALLOW_UNSIGNED_DEMO
    if not signature:
        return False
    digest = hmac.new(
        SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(digest, signature)


def append_event(event: dict[str, Any]) -> None:
    EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "razorpay-resolve-webhook-gateway"}


@app.post("/webhook/razorpay")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(default=None),
) -> dict[str, Any]:
    raw = await request.body()

    if not verify_signature(raw, x_razorpay_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    event_id = str(payload.get("id") or payload.get("event_id") or "")
    event_name = str(payload.get("event") or "unknown")

    # Razorpay webhooks can be delivered more than once. Idempotency is required.
    if event_id and event_id in _seen_event_ids:
        return {"ok": True, "duplicate": True, "event": event_name}

    if event_id:
        _seen_event_ids.add(event_id)

    append_event(
        {
            "received_event": event_name,
            "event_id": event_id,
            "payload": payload,
        }
    )

    return {"ok": True, "duplicate": False, "event": event_name}
