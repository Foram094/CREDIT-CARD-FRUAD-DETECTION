"""
Build 30-feature vectors: [Time, V1..V28, Amount] for the Kaggle creditcard model.

V1–V28 defaults come from creditcard.csv means (loaded once at startup in model_loader).
If the dataset is missing, V defaults are zeros.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.model_loader import (
    get_amount_fallback_mean,
    get_amount_high_threshold,
    get_time_fallback_mean,
    get_time_stats,
    get_v_means,
)

FEATURE_COLUMNS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Small nudges (PCA-scale features ~O(1)); keep stable for the model
_DELTA_UNUSUAL = 0.12
_DELTA_ONLINE = 0.08
_DELTA_OFFLINE = 0.06

# 0-based indices into the V1..V28 vector (different groups for location vs channel)
_UNUSUAL_V_IDX = (3, 10, 16)  # V4, V11, V17
_ONLINE_V_IDX = (4, 13, 20)  # V5, V14, V21
_OFFLINE_V_IDX = (2, 8, 14)  # V3, V9, V15


def get_feature_means() -> dict[str, float]:
    """Full column means for filling missing CSV columns (Time, V1..V28, Amount)."""
    v = get_v_means()
    means: dict[str, float] = {
        "Time": get_time_fallback_mean(),
        "Amount": get_amount_fallback_mean(),
    }
    for i in range(28):
        means[f"V{i + 1}"] = float(v[i])
    return means


def _normalize_time(raw: float | str | None) -> float:
    if raw is None:
        return get_time_fallback_mean()
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return get_time_fallback_mean()
    return float(np.clip(t, 0.0, 200_000.0))


def _normalize_amount(raw: float | str | None) -> float:
    if raw is None:
        return get_amount_fallback_mean()
    try:
        a = float(raw)
    except (TypeError, ValueError):
        return get_amount_fallback_mean()
    return float(max(0.0, a))


def _apply_v_nudges(
    v: np.ndarray,
    location: str | None,
    transaction_type: str | None,
) -> np.ndarray:
    """Copy of V vector with small location / channel adjustments."""
    out = np.array(v, dtype=np.float64, copy=True)

    loc = (location or "normal").strip().lower()
    if loc in ("unusual", "u"):
        for i in _UNUSUAL_V_IDX:
            out[i] += _DELTA_UNUSUAL

    tx = (transaction_type or "online").strip().lower()
    if tx in ("online", "on", "web"):
        for i in _ONLINE_V_IDX:
            out[i] += _DELTA_ONLINE
    elif tx in ("offline", "off", "in-person", "in_person", "store"):
        for i in _OFFLINE_V_IDX:
            out[i] -= _DELTA_OFFLINE

    return out


def simplified_to_feature_row(
    amount: float | str | None,
    time_value: float | str | None,
    location: str | None,
    transaction_type: str | None,
) -> np.ndarray:
    """
    Map UI fields to shape (1, 30): [Time, V1..V28, Amount].
    V1–V28 start as dataset means (or zeros), then nudged by location / type.
    """
    t = _normalize_time(time_value)
    a = _normalize_amount(amount)
    v = get_v_means()
    v = _apply_v_nudges(v, location, transaction_type)

    row = np.concatenate([[t], v, [a]]).astype(np.float64, copy=False)
    return row.reshape(1, 30)


def dynamic_values_to_feature_row(values: dict[str, Any]) -> np.ndarray:
    """
    Map user-supplied feature values (any subset of Time, V1..V28, Amount) to shape (1, 30).
    Missing columns use dataset means from model_loader.
    """
    means = get_feature_means()
    row_dict = {c: float(means[c]) for c in FEATURE_COLUMNS}
    for k, v in values.items():
        if k not in row_dict:
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                row_dict[k] = fv
        except (TypeError, ValueError):
            pass
    ordered = [row_dict[c] for c in FEATURE_COLUMNS]
    return np.array([ordered], dtype=np.float64)


def build_behavior_factor_notes(
    values: dict[str, Any],
    selected: list[str],
    means: dict[str, float],
    threshold: float = 2.5,
) -> list[str]:
    """Generic wording only — never expose raw V names to the user."""
    for name in selected:
        if not name.startswith("V") or name not in values:
            continue
        try:
            v = float(values[name])
            m = float(means.get(name, 0.0))
            if math.isfinite(v) and abs(v - m) > threshold:
                return [
                    "One or more internal risk/pattern signals in the row differ strongly from typical training data."
                ]
        except (TypeError, ValueError):
            continue
    return []


def dataframe_to_model_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Align CSV rows to FEATURE_COLUMNS.

    * Full Kaggle-style rows (Time, V1..V28, Amount) are used directly where present.
    * Minimal CSV (e.g. only Time + Amount): missing V1–V28 filled from dataset means (or zeros).
    * Column ``Class`` is ignored for features (not in FEATURE_COLUMNS).
    """
    means = get_feature_means()
    warnings: list[str] = []
    missing_cols: list[str] = []
    out = pd.DataFrame(index=df.index)

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = means[col]
            missing_cols.append(col)

    if missing_cols:
        if len(missing_cols) <= 5:
            for col in missing_cols:
                warnings.append(f"Column '{col}' was missing; filled with default mean.")
        else:
            sample = ", ".join(missing_cols[:4])
            warnings.append(
                f"{len(missing_cols)} columns missing ({sample}, …); filled with dataset means (or zeros if no CSV)."
            )

    out = out.fillna(pd.Series(means))
    return out[FEATURE_COLUMNS].to_numpy(dtype=np.float64), warnings


def risk_level_from_score(risk_pct: float) -> str:
    """Map model fraud probability (0–100) to Low / Medium / High."""
    if risk_pct > 70:
        return "High"
    if risk_pct >= 40:
        return "Medium"
    return "Low"


def suggested_actions_for_level(level: str) -> list[str]:
    """Practical next steps (interpretation only — not automatic bank actions)."""
    if level == "High":
        return [
            "Block or freeze the card/account for outgoing payments until the customer confirms the activity.",
            "Contact your bank immediately (use the number on the back of the card — not links from messages).",
            "Request a temporary freeze on the account if the bank offers it while you investigate.",
            "Report suspected fraud through your bank’s official fraud channel or app.",
        ]
    if level == "Medium":
        return [
            "Verify this transaction with the customer (amount and whether they recognize the charge).",
            "Monitor account activity closely for the next few days for repeat or related charges.",
        ]
    return [
        "No immediate action required at this risk level — continue routine monitoring.",
    ]


def build_risk_factors(
    amount: float | None,
    time_val: float | None,
    location: str | None,
    transaction_type: str | None,
) -> list[str]:
    """Discrete risk/context factors (same signals as before, as a list)."""
    factors: list[str] = []
    p95 = get_amount_high_threshold()
    t_mean, t_std = get_time_stats()

    if amount is not None and math.isfinite(float(amount)):
        if float(amount) >= p95:
            factors.append(
                "Transaction amount is elevated versus typical values in the reference data."
            )
        elif float(amount) <= 0:
            factors.append("Amount was missing or zero; defaults were applied for scoring.")

    if time_val is not None and math.isfinite(float(time_val)):
        time_val = float(time_val)
        if t_std > 0 and abs(time_val - t_mean) > 2.0 * t_std:
            factors.append("Timing is far from the usual distribution (unusual time of activity).")
        elif time_val > 170_000 or (time_val > 0 and time_val < 60):
            factors.append("Time index falls in a less common range for this dataset.")

    loc = (location or "").strip().lower()
    if loc in ("unusual", "u"):
        factors.append("Location marked as unusual — risk profile adjusted accordingly.")

    tx = (transaction_type or "").strip().lower()
    if tx in ("online", "on", "web"):
        factors.append("Online channel context was applied to the feature profile.")
    elif tx in ("offline", "off", "in-person", "in_person", "store"):
        factors.append("In-person / offline channel context was applied to the feature profile.")

    if not factors:
        factors.append("No strong contextual flags from inputs — assessment relies mainly on the model.")

    return factors


def build_smart_insight(
    factors: list[str],
    risk_pct: float,
    result_label: str,
) -> str:
    """One concise headline for decision support."""
    level = risk_level_from_score(risk_pct)
    if level == "High" or result_label == "Fraud":
        if any("amount" in f.lower() for f in factors):
            return "High fraud probability with a notable amount pattern — treat as suspicious."
        if any("time" in f.lower() or "timing" in f.lower() for f in factors):
            return "High fraud probability with unusual timing relative to normal activity."
        return "Model indicates elevated fraud risk; pattern departs from typical safe behavior."

    if level == "Medium":
        return "Mixed signals: not clearly safe or clearly fraudulent — verification is appropriate."

    if any("amount" in f.lower() and "elevated" in f.lower() for f in factors):
        return "Score is lower, but the amount is still worth a quick sanity check."

    return "Transaction aligns with a low-risk profile based on this model and inputs."


def build_explanation(
    result_label: str,
    fraud_probability: float,
    amount: float,
    time_val: float,
    location: str | None,
    transaction_type: str | None,
) -> str:
    """Legacy full paragraph (includes factors + probability + classification)."""
    factors = build_risk_factors(amount, time_val, location, transaction_type)
    tail = [
        f"Estimated fraud probability: {fraud_probability * 100:.1f}%.",
        (
            "The model classifies this case as fraud at its decision threshold."
            if result_label == "Fraud"
            else "The model classifies this case as safe at its decision threshold."
        ),
    ]
    return " ".join(factors + tail)


def enrich_interpretation(
    risk_pct: float,
    result_label: str,
    amount: float | None,
    time_val: float | None,
    location: str | None = None,
    transaction_type: str | None = None,
    extra_factors: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Post-prediction guidance: levels, factors, actions, insight (no model change)."""
    factors = build_risk_factors(amount, time_val, location, transaction_type)
    if extra_factors:
        generic = "No strong contextual flags from inputs — assessment relies mainly on the model."
        if len(factors) == 1 and factors[0] == generic:
            factors = []
        for ef in extra_factors:
            if ef not in factors:
                factors.append(ef)
        if not factors:
            factors = list(extra_factors)
    level = risk_level_from_score(risk_pct)
    actions = suggested_actions_for_level(level)
    insight = build_smart_insight(factors, risk_pct, result_label)
    explanation = build_explanation(
        result_label,
        risk_pct / 100.0,
        amount if amount is not None else get_amount_fallback_mean(),
        time_val if time_val is not None else get_time_fallback_mean(),
        location,
        transaction_type,
    )
    if extra_factors:
        explanation = " ".join(extra_factors) + " " + explanation
    return {
        "risk_level": level,
        "factors": factors,
        "suggested_actions": actions,
        "insight": insight,
        "explanation": explanation,
    }


def prediction_response_dict_dynamic(
    model: Any,
    X: np.ndarray,
    values: dict[str, Any],
    selected_features: list[str],
) -> dict[str, Any]:
    """Same outputs as prediction_response_dict, for dynamic user-selected features."""
    proba = model.predict_proba(X)
    fraud_col = 1 if proba.shape[1] > 1 else 0
    fraud_p = float(proba[0, fraud_col])
    pred = int(model.predict(X)[0])
    result = "Fraud" if pred == 1 else "Safe"
    risk_pct = round(fraud_p * 100, 2)

    means = get_feature_means()
    extras = build_behavior_factor_notes(values, selected_features, means)
    extra = enrich_interpretation(
        risk_pct,
        result,
        float(X[0, -1]),
        float(X[0, 0]),
        None,
        None,
        extra_factors=extras if extras else None,
    )

    return {
        "result": result,
        "prediction": result,
        "risk_score": risk_pct,
        "risk_level": extra["risk_level"],
        "factors": extra["factors"],
        "suggested_actions": extra["suggested_actions"],
        "insight": extra["insight"],
        "explanation": extra["explanation"],
    }


def prediction_response_dict(
    model: Any,
    X: np.ndarray,
    amount_for_text: float,
    time_for_text: float,
    location: str | None,
    transaction_type: str | None,
) -> dict[str, Any]:
    proba = model.predict_proba(X)
    fraud_col = 1 if proba.shape[1] > 1 else 0
    fraud_p = float(proba[0, fraud_col])
    pred = int(model.predict(X)[0])
    result = "Fraud" if pred == 1 else "Safe"
    risk_pct = round(fraud_p * 100, 2)

    extra = enrich_interpretation(
        risk_pct,
        result,
        amount_for_text,
        time_for_text,
        location,
        transaction_type,
    )

    return {
        "result": result,
        "prediction": result,
        "risk_score": risk_pct,
        "risk_level": extra["risk_level"],
        "factors": extra["factors"],
        "suggested_actions": extra["suggested_actions"],
        "insight": extra["insight"],
        "explanation": extra["explanation"],
    }
