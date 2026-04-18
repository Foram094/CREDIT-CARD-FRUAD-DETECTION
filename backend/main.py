"""
FastAPI app: load creditcard.csv stats, fraud_model.pkl (or auto-train), expose /predict and /predict-file.
Run from project root: uvicorn backend.main:app --reload
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.feature_selector import analyze_and_save_feature_config, get_feature_config
from backend.utils import (
    FEATURE_COLUMNS,
    build_behavior_factor_notes,
    get_feature_means,
)
from backend.model_loader import (
    CSV_PATH,
    dataset_loaded_ok,
    get_model,
    get_model_path,
    is_model_available,
    load_kaggle_creditcard_dataset,
    load_model,
)

MODEL_UNAVAILABLE_MSG = (
    "Model not available. Please ensure dataset is present to auto-train, "
    "or add fraud_model.pkl to the project root."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ds_info = load_kaggle_creditcard_dataset()
        app.state.dataset_info = ds_info
    except Exception as e:
        app.state.dataset_info = {"creditcard_csv": False, "error": str(e)}

    try:
        load_model()
    except Exception as e:
        # load_model is defensive; this is only unexpected errors
        print(f"⚠️ Unexpected error during model setup: {e}", flush=True)

    app.state.model_message = None if is_model_available() else MODEL_UNAVAILABLE_MSG

    df = None
    if CSV_PATH.is_file():
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            pass
    try:
        app.state.feature_config = analyze_and_save_feature_config(get_model(), df)
    except Exception as e:
        print(f"⚠️ Feature analysis skipped: {e}", flush=True)
        app.state.feature_config = get_feature_config()

    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Dynamic keys must match GET /feature-config → selected_features (Time, Amount, V*, …)."""

    values: dict[str, Any] = Field(default_factory=dict)


def _model_unavailable_response():
    return JSONResponse(
        status_code=503,
        content={"error": MODEL_UNAVAILABLE_MSG},
    )


def _import_utils_predict():
    from backend.utils import (
        dataframe_to_model_matrix,
        dynamic_values_to_feature_row,
        enrich_interpretation,
        prediction_response_dict_dynamic,
    )

    return dynamic_values_to_feature_row, prediction_response_dict_dynamic, dataframe_to_model_matrix, enrich_interpretation


@app.get("/health")
def health():
    mp = get_model_path()
    model_loaded = is_model_available()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": str(mp) if mp is not None else None,
        "dataset_loaded": dataset_loaded_ok(),
        "ok": True,
        "detail": None if model_loaded else getattr(app.state, "model_message", MODEL_UNAVAILABLE_MSG),
        "model_detail": None if model_loaded else getattr(app.state, "model_message", MODEL_UNAVAILABLE_MSG),
        "creditcard_csv_loaded": dataset_loaded_ok(),
        "dataset": getattr(app.state, "dataset_info", {}),
        "feature_inputs": len(get_feature_config().get("selected_features", [])),
    }


@app.get("/feature-config")
def feature_config():
    """UI uses this to render inputs (labels, types, ranges) — adapts to dataset + model."""
    return get_feature_config()


@app.post("/predict")
def predict(body: PredictRequest):
    model = get_model()
    if model is None:
        return _model_unavailable_response()

    dynamic_values_to_feature_row, prediction_response_dict_dynamic, *_ = _import_utils_predict()

    cfg = get_feature_config()
    allowed = cfg.get("selected_features", [])
    allowed_set = set(allowed)
    raw = body.values or {}
    values: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in allowed_set:
            continue
        if v is None or v == "":
            continue
        try:
            values[k] = float(v)
        except (TypeError, ValueError):
            continue

    X = dynamic_values_to_feature_row(values)
    return prediction_response_dict_dynamic(model, X, values, allowed)


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        return _model_unavailable_response()

    *_, dataframe_to_model_matrix, enrich_interpretation = _import_utils_predict()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        text = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {e}",
        ) from e

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")

    drop_extra = [c for c in df.columns if c == "Class"]
    if drop_extra:
        df = df.drop(columns=drop_extra)

    try:
        X, fill_messages = dataframe_to_model_matrix(df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not build feature matrix: {e}",
        ) from e

    try:
        preds = model.predict(X)
        proba = model.predict_proba(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed (check feature count matches training): {e}",
        ) from e

    fraud_col = 1 if proba.shape[1] > 1 else 0
    cfg_features = get_feature_config().get("selected_features", [])
    means = get_feature_means()

    rows_out: list[dict[str, Any]] = []
    for i in range(len(df)):
        fp = float(proba[i, fraud_col])
        label = "Fraud" if int(preds[i]) == 1 else "Safe"
        risk_pct = round(fp * 100, 2)

        amt = None
        tval = None
        if "Amount" in df.columns and pd.notna(df["Amount"].iloc[i]):
            amt = float(df["Amount"].iloc[i])
        if "Time" in df.columns and pd.notna(df["Time"].iloc[i]):
            tval = float(df["Time"].iloc[i])

        row_values = {FEATURE_COLUMNS[j]: float(X[i, j]) for j in range(len(FEATURE_COLUMNS))}
        extras = build_behavior_factor_notes(row_values, cfg_features, means)
        interp = enrich_interpretation(
            risk_pct,
            label,
            amt,
            tval,
            None,
            None,
            extra_factors=extras if extras else None,
        )

        rows_out.append(
            {
                "row": i + 1,
                "prediction": label,
                "result": label,
                "risk_score": risk_pct,
                "risk_level": interp["risk_level"],
                "factors": interp["factors"],
                "suggested_actions": interp["suggested_actions"],
                "insight": interp["insight"],
                "explanation": interp["explanation"],
            }
        )

    return {
        "count": len(rows_out),
        "warnings": fill_messages[:20],
        "predictions": rows_out,
    }
