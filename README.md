/*
## Dataset Setup

This project uses the Kaggle Credit Card Fraud Detection dataset.

Due to size constraints, the dataset is not included in the repository.

Steps:

1. Download dataset from Kaggle
2. Place `creditcard.csv` in project root folder

The system will still run without it, but dataset improves:

* feature selection
* default values
* input ranges






*/


# Fraud Guard — local fraud detection demo

Stack: **FastAPI** backend, **HTML/CSS/JS** frontend, **scikit-learn** model (`fraud_model.pkl` or auto-train from `creditcard.csv`).

## Run the backend (VS Code terminal)

From the **project root** (folder that contains `backend/` and `frontend/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

API: `http://127.0.0.1:8000` — docs at `/docs`.

### Adaptive inputs

- **`GET /feature-config`** — JSON describing which fields the UI should show (always **Time** + **Amount** plus the top **4** anonymized `V*` columns by importance). Importance comes from the loaded model’s coefficients (or a short-lived logistic fit on `creditcard.csv` if needed). The file `backend/feature_config.json` is rewritten on each API startup.
- **`POST /predict`** — body shape: `{ "values": { "Time": 50000, "Amount": 99.5, "V14": 0.2, ... } }`. Any feature not sent is filled with dataset means on the server (full 30-feature vector unchanged for the model).

## Run the frontend

Open `frontend/index.html` in a browser (or use Live Server). The UI calls the API at `http://127.0.0.1:8000`.

## Sample CSV files (`sample_data/`)

Use these with **Upload & predict** to demo `/predict-file` without sharing the full Kaggle file.

| File | Purpose | Expected behavior |
|------|---------|---------------------|
| `sample_full_format.csv` | Full Kaggle-style columns: `Time`, `V1`–`V28`, `Amount` (25 rows). Mix of “normal” and “spikier” PCA-style rows. | Exercises the path where all features are present; scores vary by row. |
| `sample_simple_input.csv` | Only `Time` and `Amount` (~17 rows). Typical values plus a few high amounts and odd times. | Missing `V*` columns are filled from dataset means (or zeros if no `creditcard.csv`); good for quick tests. |
| `sample_high_risk.csv` | Large **Amount** values and extreme / odd **Time** values. | Tends to push the model toward **higher fraud probability** (not guaranteed — depends on your trained weights). |
| `sample_low_risk.csv` | Small amounts and mid-range times. | Tends toward **lower** scores / **Safe**-leaning predictions. |

**Tip:** After a fresh clone, add `creditcard.csv` and/or `fraud_model.pkl` to the project root so auto-train and feature defaults match the real dataset.

## API response (manual `/predict`)

Includes `prediction` / `result`, `risk_score`, `risk_level`, `factors`, `suggested_actions`, `insight`, and `explanation`.

## API response (`/predict-file`)

Each element in `predictions` includes the same enrichment as manual mode: `prediction`, `risk_score`, `risk_level`, `factors`, `suggested_actions`, `insight`, `explanation`, plus `row`. The UI table summarizes **key factor**; use the JSON for full actions per row.

## Help resources (India)

The frontend **Need help?** section links to **cybercrime.gov.in** and notes **RBI 14440** — verify official sources for the latest numbers and hours.
