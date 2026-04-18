"""
Load the trained fraud model and Kaggle creditcard.csv statistics at startup.
If fraud_model.pkl is missing, optionally train a simple model from creditcard.csv.
"""

from __future__ import annotations

import pickle
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Project root = directory that contains backend/ and (expected) fraud_model.pkl
_BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _BACKEND_DIR.parent

MODEL_FILENAME = "fraud_model.pkl"
CSV_FILENAME = "creditcard.csv"
CSV_PATH = PROJECT_ROOT / CSV_FILENAME

_model: Any | None = None
_model_file_path: Path | None = None
_auto_train_attempted: bool = False

# Populated by load_kaggle_creditcard_dataset()
_V_MEANS: np.ndarray | None = None
_TIME_FALLBACK: float = 94_813.0
_AMOUNT_FALLBACK: float = 88.35
_AMOUNT_P95: float = 180.0
_TIME_MEAN: float = 94_813.0
_TIME_STD: float = 47_482.0
_DATASET_FOUND: bool = False


def _candidate_model_paths() -> list[Path]:
    name = MODEL_FILENAME
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        rp = p.resolve()
        key = str(rp)
        if key not in seen:
            seen.add(key)
            out.append(rp)

    add(PROJECT_ROOT / name)
    add(Path.cwd() / name)
    add(_BACKEND_DIR / name)

    here = Path.cwd().resolve()
    for _ in range(5):
        add(here / name)
        parent = here.parent
        if parent == here:
            break
        here = parent

    return out


def _resolve_model_path() -> Path | None:
    for p in _candidate_model_paths():
        if p.is_file():
            return p
    return None


MODEL_PATH = PROJECT_ROOT / MODEL_FILENAME


def get_model_path() -> Optional[Path]:
    """Resolved path to fraud_model.pkl if it exists on disk or was loaded; else None."""
    if _model_file_path is not None:
        return _model_file_path
    return _resolve_model_path()


def is_model_available() -> bool:
    return _model is not None


def train_and_save_model_if_missing() -> Optional[Path]:
    """
    If no model file exists, train LogisticRegression on creditcard.csv and save to project root.
    Returns path to saved file on success, None if skipped or failed (logs reason).
    """
    if _resolve_model_path() is not None:
        return _resolve_model_path()

    if not CSV_PATH.is_file():
        print(
            "⚠️ creditcard.csv not found in project root — cannot auto-train. "
            "Add creditcard.csv or place fraud_model.pkl next to backend/.",
            flush=True,
        )
        return None

    try:
        from sklearn.linear_model import LogisticRegression

        df = pd.read_csv(CSV_PATH)
        if "Class" not in df.columns:
            print("⚠️ creditcard.csv has no 'Class' column — cannot auto-train.", flush=True)
            return None

        feature_df = df.drop(columns=["Class"])
        X = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)

        model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        print("Training fallback LogisticRegression on creditcard.csv (may take a minute)...", flush=True)
        model.fit(X.to_numpy(dtype=np.float64), y.to_numpy())

        save_path = (PROJECT_ROOT / MODEL_FILENAME).resolve()
        try:
            import joblib

            joblib.dump(model, save_path)
        except Exception:
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

        print(f"Model auto-trained and saved to: {save_path}", flush=True)
        return save_path
    except Exception as e:
        print(f"⚠️ Auto-train failed: {e}", flush=True)
        traceback.print_exc()
        return None


def load_model() -> Optional[Any]:
    """
    Load or auto-train the model. Never raises for missing file — returns None if unavailable.
    """
    global _model, _model_file_path, _auto_train_attempted

    if _model is not None:
        return _model

    path = _resolve_model_path()
    if path is None and not _auto_train_attempted:
        _auto_train_attempted = True
        print("⚠️ Model not found. Attempting auto-train...", flush=True)
        train_and_save_model_if_missing()
        path = _resolve_model_path()

    if path is None:
        print(
            "❌ Model still not available. API will run but predictions are disabled. "
            "Add creditcard.csv (for auto-train) or fraud_model.pkl in the project root.",
            flush=True,
        )
        _model = None
        _model_file_path = None
        return None

    try:
        import joblib

        _model = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            _model = pickle.load(f)

    if not hasattr(_model, "predict") or not hasattr(_model, "predict_proba"):
        print(f"❌ Invalid model object at {path} (missing predict / predict_proba).", flush=True)
        _model = None
        _model_file_path = None
        return None

    _model_file_path = path.resolve()
    print(f"Model loaded successfully from: {_model_file_path}", flush=True)
    return _model


def get_model() -> Optional[Any]:
    """In-memory model, or None if not loaded."""
    return _model


def load_kaggle_creditcard_dataset() -> dict[str, Any]:
    """
    Load creditcard.csv once: compute mean for V1–V28 and summary stats for explanations.
    If the file is missing, use zeros for V means and static fallbacks for Time/Amount.
    """
    global _V_MEANS, _TIME_FALLBACK, _AMOUNT_FALLBACK, _AMOUNT_P95, _TIME_MEAN, _TIME_STD, _DATASET_FOUND

    v_names = [f"V{i}" for i in range(1, 29)]
    _V_MEANS = np.zeros(28, dtype=np.float64)
    _DATASET_FOUND = False

    if not CSV_PATH.is_file():
        _TIME_FALLBACK = 94_813.0
        _AMOUNT_FALLBACK = 88.35
        _AMOUNT_P95 = 180.0
        _TIME_MEAN = _TIME_FALLBACK
        _TIME_STD = 47_482.0
        return {
            "creditcard_csv": False,
            "path": str(CSV_PATH),
            "v_means_source": "zeros (file not found)",
        }

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        _TIME_FALLBACK = 94_813.0
        _AMOUNT_FALLBACK = 88.35
        _AMOUNT_P95 = 180.0
        _TIME_MEAN = _TIME_FALLBACK
        _TIME_STD = 47_482.0
        return {
            "creditcard_csv": False,
            "path": str(CSV_PATH),
            "v_means_source": "zeros (read failed)",
        }

    _DATASET_FOUND = True

    for i, name in enumerate(v_names):
        if name in df.columns:
            _V_MEANS[i] = float(pd.to_numeric(df[name], errors="coerce").mean())

    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce")
        _TIME_FALLBACK = float(t.mean())
        _TIME_MEAN = _TIME_FALLBACK
        _TIME_STD = float(t.std()) if float(t.std()) > 1.0 else 47_482.0
    else:
        _TIME_MEAN = _TIME_FALLBACK
        _TIME_STD = 47_482.0

    if "Amount" in df.columns:
        a = pd.to_numeric(df["Amount"], errors="coerce")
        _AMOUNT_FALLBACK = float(a.mean())
        _AMOUNT_P95 = float(a.quantile(0.95))
    else:
        _AMOUNT_FALLBACK = 88.35
        _AMOUNT_P95 = 180.0

    return {
        "creditcard_csv": True,
        "path": str(CSV_PATH),
        "rows": len(df),
        "v_means_source": "creditcard.csv column means",
    }


def get_v_means() -> np.ndarray:
    if _V_MEANS is None:
        return np.zeros(28, dtype=np.float64)
    return _V_MEANS.copy()


def get_time_fallback_mean() -> float:
    return float(_TIME_FALLBACK)


def get_amount_fallback_mean() -> float:
    return float(_AMOUNT_FALLBACK)


def get_amount_high_threshold() -> float:
    return float(_AMOUNT_P95)


def get_time_stats() -> tuple[float, float]:
    return float(_TIME_MEAN), float(_TIME_STD)


def dataset_loaded_ok() -> bool:
    return bool(_DATASET_FOUND)
