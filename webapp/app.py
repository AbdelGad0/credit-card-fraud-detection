"""
app.py — Standard web application (Flask) for the fraud-detection model.

A plain HTML/CSS/JS frontend talks to this small JSON API. The model weights
and the train-fitted RobustScaler are embedded once at startup via
`predictor.FraudPredictor` (no re-training).

Endpoints
---------
GET  /                        -> landing page (templates/index.html)
GET  /api/glossary            -> column glossary + research note (JSON)
POST /api/predict/upload      -> multipart CSV/XLSX batch classification
POST /api/predict/sample      -> classification of a real held-out sample
POST /api/predict/form        -> single-row manual check (JSON body)

Run from the project root:
    C:\\Users\\abdel\\anaconda3\\python.exe webapp/app.py --port 8000
"""

import base64
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

from predictor import FraudPredictor, REQUIRED_COLS  # noqa: E402
from schema import GLOSSARY, SUMMARY_NOTE, SUMMARY_NOTE_EN, display  # noqa: E402

app = Flask(__name__)

# Load the model + scaler ONCE at startup (embedded weights, never re-trained)
predictor = FraudPredictor()

# ---------------------------------------------------------------------------
# Constants for the response payload (lean JSON, download stays complete)
# ---------------------------------------------------------------------------
TABLE_ROW_LIMIT = 200      # rows sent for on-screen tables
CSV_ROW_LIMIT = 50_000     # rows embedded as downloadable CSV


def _to_scalar(v):
    """Convert numpy/pandas scalars to native Python for JSON serialization."""
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(round(v, 6))
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _rows_to_records(df: pd.DataFrame) -> list:
    return [
        {k: _to_scalar(v) for k, v in row.items()}
        for _, row in df.iterrows()
    ]


def _chart_hist_b64(df: pd.DataFrame, threshold: float) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(df["Fraud_Probability"], bins=40, color="#4C72B0")
    ax.axvline(threshold, color="#C44E52", ls="--", lw=1.5,
               label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Fraud probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_cat_b64(df: pd.DataFrame) -> str:
    counts = df["Prediction"].value_counts().reindex(
        ["Fraud", "Legitimate"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(["Fraud", "Legitimate"], counts.values,
           color=["#C44E52", "#4C72B0"])
    ax.set_ylabel("Count")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _score_payload(result, threshold: float) -> dict:
    """Build the JSON response for a scored batch (friendly column names)."""
    disp = display(result.table)
    flagged = display(result.flagged)
    if "Fraud_Label" in flagged.columns:
        flagged = flagged.drop(columns=["Fraud_Label"])
    order = ["Prediction", "Fraud_Probability"] + [
        c for c in flagged.columns
        if c not in ("Prediction", "Fraud_Probability")]
    flagged = flagged[order]

    payload = {
        "meta": {
            "total": int(len(result.table)),
            "fraud": int(result.fraud_count),
            "legit": int(result.legit_count),
            "threshold": float(threshold),
            "tables_truncated": len(disp) > TABLE_ROW_LIMIT,
            "csv_truncated": len(disp) > CSV_ROW_LIMIT,
            "default_threshold": float(predictor.default_threshold),
        },
        "metrics": result.metrics,
        "chart_hist": _chart_hist_b64(result.table, threshold),
        "chart_cat": _chart_cat_b64(result.table),
        "flagged": _rows_to_records(flagged.head(TABLE_ROW_LIMIT)),
        "results": _rows_to_records(disp.head(TABLE_ROW_LIMIT)),
        "columns": list(disp.columns),
    }

    head = disp.head(CSV_ROW_LIMIT)
    payload["csv_b64"] = base64.b64encode(
        head.to_csv(index=False).encode("utf-8-sig")).decode()
    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/glossary")
def api_glossary():
    return jsonify({
        "note_ar": SUMMARY_NOTE,
        "note_en": SUMMARY_NOTE_EN,
        "rows": GLOSSARY,
        "required": REQUIRED_COLS,
    })


@app.post("/api/predict/upload")
def api_upload():
    threshold = float(request.form.get("threshold", predictor.default_threshold))
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no_file"}), 400

    try:
        df = predictor.normalize_columns(_read_upload(f))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "unreadable", "detail": str(exc)}), 400
    return _score(df, threshold, f.filename)


@app.post("/api/predict/sample")
def api_sample():
    threshold = float(request.form.get("threshold", predictor.default_threshold))
    df = predictor.normalize_columns(
        predictor.load_raw_sample(n=5, with_ground_truth=True))
    return _score(df, threshold,
                  "sample" if request.args.get("lang") == "en" else
                  "عيّنة تجريبية من التست (5 صفوف حقيقية)")


@app.post("/api/predict/form")
def api_form():
    body = request.get_json(silent=True) or {}
    threshold = float(body.get("threshold", predictor.default_threshold))
    row = body.get("row") or {}
    try:
        df = predictor.normalize_columns(pd.DataFrame([row]))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "form_read", "detail": str(exc)}), 400

    missing = predictor._validate_columns(df)
    if missing:
        return jsonify({"error": "missing_cols", "cols": missing}), 400
    try:
        result = predictor.predict(df, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "predict", "detail": str(exc)}), 400

    r = result.table.iloc[0]
    return jsonify({
        "probability": float(round(r["Fraud_Probability"], 6)),
        "prediction": r["Prediction"],
        "threshold": float(threshold),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_upload(uploaded):
    if uploaded.filename.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    if uploaded.filename.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded, engine="openpyxl")
    raise ValueError("bad_extension")


def _score(df: pd.DataFrame, threshold: float, source_label: str):
    missing = predictor._validate_columns(df)
    if missing:
        return jsonify({"error": "missing_cols", "cols": missing}), 400
    try:
        result = predictor.predict(df, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "predict", "detail": str(exc)}), 400

    payload = _score_payload(result, threshold)
    payload["source"] = source_label
    return jsonify(payload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud-detection web app")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"Fraud-detection web app -> http://localhost:{args.port}")
    app.run(host=args.host, port=args.port)