from __future__ import annotations
import pandas as pd


def reconcile(df: pd.DataFrame, tolerance: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    out["recon_difference"] = (
        out["settlement_amount"] - out["expected_settlement"]
    )
    out["recon_status"] = out["recon_difference"].abs().le(tolerance).map(
        {True: "Matched", False: "Mismatch"}
    )
    out["recon_priority"] = out["recon_difference"].abs().apply(
        lambda x: "Critical" if x > 50 else ("Review" if x > 5 else "Normal")
    )
    return out


def reconciliation_summary(df: pd.DataFrame, tolerance: float = 1.0) -> dict[str, float]:
    checked = reconcile(df, tolerance)
    mismatches = checked["recon_status"].eq("Mismatch")
    return {
        "checked": int(len(checked)),
        "matched": int((~mismatches).sum()),
        "mismatches": int(mismatches.sum()),
        "mismatch_value": float(checked.loc[mismatches, "recon_difference"].abs().sum()),
    }
