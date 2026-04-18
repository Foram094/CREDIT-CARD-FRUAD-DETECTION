"""
Analyze creditcard-style data and pick top interactive features (Time, Amount + top V's).
Writes backend/feature_config.json at startup — adapts when dataset or model changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _BACKEND_DIR / "feature_config.json"

FEATURE_COLUMNS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MAX_USER_FEATURES = 6
NUM_BEHAVIOR_SLOTS = 4  # Time + Amount + 4 V's = 6

SIGNAL_HELP_TEXT = (
    "Derived model feature (no direct real-world meaning). "
    "Internal signals produced from transaction patterns — not a literal field like location."
)

_cached_config: dict[str, Any] | None = None


def _default_v_fallback() -> list[str]:
    return ["V14", "V17", "V12", "V10"]


def _coef_magnitude(model: Any) -> Optional[np.ndarray]:
    if not hasattr(model, "coef_"):
        return None
    c = np.asarray(model.coef_, dtype=np.float64)
    if c.ndim == 2:
        return np.mean(np.abs(c), axis=0)
    return np.abs(c.ravel())


def _importances_from_model(model: Any) -> Optional[dict[str, float]]:
    names = FEATURE_COLUMNS
    coef = _coef_magnitude(model)
    if coef is not None and len(coef) == len(names):
        return {names[i]: float(coef[i]) for i in range(len(names))}
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        if len(imp) == len(names):
            return {names[i]: float(imp[i]) for i in range(len(names))}
    return None


def _importances_from_temp_lr(df: pd.DataFrame) -> dict[str, float]:
    if "Class" not in df.columns:
        return {n: 1.0 for n in FEATURE_COLUMNS}
    if not all(c in df.columns for c in FEATURE_COLUMNS):
        return {n: 1.0 for n in FEATURE_COLUMNS}

    sub = df[FEATURE_COLUMNS + ["Class"]].copy()
    for c in FEATURE_COLUMNS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 200:
        return {n: 1.0 for n in FEATURE_COLUMNS}

    try:
        from sklearn.linear_model import LogisticRegression

        X = sub[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = sub["Class"].to_numpy(dtype=np.int64)
        lr = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        lr.fit(X, y)
        coef = np.asarray(lr.coef_, dtype=np.float64)
        mag = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef.ravel())
        return {FEATURE_COLUMNS[i]: float(mag[i]) for i in range(len(FEATURE_COLUMNS))}
    except Exception:
        return {n: 1.0 for n in FEATURE_COLUMNS}


def _pick_top_v_scores(scores: dict[str, float]) -> list[str]:
    v_names = [f"V{i}" for i in range(1, 29)]
    ranked = sorted(
        v_names,
        key=lambda x: (-scores.get(x, 0.0), int(x[1:])),
    )
    return ranked[:NUM_BEHAVIOR_SLOTS]


def _ranges_from_df(df: pd.DataFrame | None, selected: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in selected:
        if df is not None and name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce").dropna()
            if len(s) > 50:
                lo = float(s.quantile(0.01))
                hi = float(s.quantile(0.99))
                if lo >= hi:
                    lo, hi = float(s.min()), float(s.max())
                mid = float(s.median())
                pad = (hi - lo) * 0.05 + 1e-6
                out[name] = {
                    "min": max(lo - pad, lo - 1),
                    "max": hi + pad,
                    "default": mid,
                }
                continue
        if name == "Time":
            out[name] = {"min": 0.0, "max": 200_000.0, "default": 94_813.0}
        elif name == "Amount":
            out[name] = {"min": 0.0, "max": 25_000.0, "default": 88.0}
        else:
            out[name] = {"min": -10.0, "max": 10.0, "default": 0.0}
    return out


def _build_config_dict(selected: list[str], ranges: dict[str, dict[str, float]]) -> dict[str, Any]:
    feature_types: dict[str, str] = {}
    behavior_labels: dict[str, str] = {}
    v_only = [f for f in selected if f not in ("Time", "Amount")]
    for i, f in enumerate(v_only):
        feature_types[f] = "behavior"
        if i < 2:
            behavior_labels[f] = f"Risk signal {i + 1}"
        else:
            behavior_labels[f] = f"Pattern signal {i - 1}"

    for f in selected:
        if f in ("Time", "Amount"):
            feature_types[f] = "number"

    return {
        "selected_features": selected,
        "feature_types": feature_types,
        "behavior_labels": behavior_labels,
        "ranges": ranges,
        "signal_help": SIGNAL_HELP_TEXT,
    }


def analyze_and_save_feature_config(
    model: Any | None,
    df: pd.DataFrame | None,
) -> dict[str, Any]:
    """
    Rank V1–V28 by importance (loaded model or temporary LR on df), always keep Time + Amount.
    Persist to feature_config.json and refresh cache.
    """
    global _cached_config

    scores: dict[str, float] = {n: 1.0 for n in FEATURE_COLUMNS}

    imp_model = _importances_from_model(model) if model is not None else None
    if imp_model:
        scores = imp_model
    elif df is not None:
        scores = _importances_from_temp_lr(df)

    top_v = _pick_top_v_scores(scores)
    selected = ["Time", "Amount"] + top_v
    if len(selected) > MAX_USER_FEATURES:
        selected = selected[:MAX_USER_FEATURES]

    ranges = _ranges_from_df(df, selected)
    config = _build_config_dict(selected, ranges)

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Feature config saved: {len(selected)} inputs → {CONFIG_PATH.name}", flush=True)
    except Exception as e:
        print(f"⚠️ Could not write {CONFIG_PATH}: {e}", flush=True)

    _cached_config = config
    return config


def get_feature_config() -> dict[str, Any]:
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                _cached_config = json.load(f)
            if "signal_help" not in _cached_config:
                _cached_config["signal_help"] = SIGNAL_HELP_TEXT
            return _cached_config
        except Exception:
            pass
    selected = ["Time", "Amount"] + _default_v_fallback()
    _cached_config = _build_config_dict(
        selected[:MAX_USER_FEATURES],
        _ranges_from_df(None, selected[:MAX_USER_FEATURES]),
    )
    return _cached_config


def clear_feature_config_cache() -> None:
    global _cached_config
    _cached_config = None
