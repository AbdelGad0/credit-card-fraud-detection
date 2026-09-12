"""
predictor.py — Production-style prediction service for the fraud-detection model.

Encapsulates the FULL inference pipeline so the web app (and any future API)
only deals with raw incoming data:

    raw rows (Time, V1..V28, Amount)
        │
        ▼
  [1] Column validation + subset to the 30 model features
        ▼
  [2] RobustScaler.transform on [Amount, Time]   (scaler was fit on TRAIN only)
        ▼
  [3] LightGBM.predict_proba → P(Fraud)
        ▼
  [4] Prediction label via an adjustable threshold (default = F1-optimal)
        ▼
  labeled DataFrame + optional evaluation metrics (if ground truth supplied)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import sys

# Allow `python deployment/predictor.py` from the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ctypes  # noqa: E402


def _preload_libgomp() -> None:
    """Expose OpenMP before lightgbm/sklearn dlopen it at runtime.

    Serverless runtimes (Vercel, AWS Lambda) ship Python without libgomp, so
    a vendored copy is loaded into the process first; later dlopen() calls for
    ``libgomp.so.1`` are then resolved from the already-loaded object.
    """
    for lib in (ROOT / "vendor" / "libgomp.so.1", ROOT / "vendor" / "libgomp.so.1.0.0"):
        if lib.exists():
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_preload_libgomp()

import config as cfg  # noqa: E402

REQUIRED_COLS = cfg.V_COLS + [cfg.TIME_COL, cfg.AMOUNT_COL]   # 30 raw columns
GROUND_TRUTH_COL = cfg.TARGET_COL                             # optional "Class"


@dataclass
class PredictionResult:
    """Everything the UI needs after scoring a batch of new data."""
    table: pd.DataFrame                       # original + probability + prediction
    fraud_count: int
    legit_count: int
    metrics: Optional[dict]                   # set when ground-truth Class is present
    flagged: pd.DataFrame                     # rows predicted as fraud


class FraudPredictor:
    """Loads the trained model + scaler once and reuses it for many requests."""

    def __init__(self,
                 model_path: Path = cfg.MODEL_DIR / "LightGBM_smote.pkl",
                 scaler_path: Path = cfg.DATA_DIR / "scaler.pkl",
                 default_threshold: Optional[float] = None,
                 random_state: int = cfg.RANDOM_STATE):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        # F1-optimal threshold discovered on the held-out test set.
        self.default_threshold = default_threshold or self._load_optimal_threshold()
        self.random_state = random_state

    # ------------------------------------------------------------------
    @staticmethod
    def _load_optimal_threshold() -> float:
        """Read the F1-optimal threshold saved by 05_evaluation.py."""
        best_file = cfg.RESULTS_DIR / "best_model.json"
        if best_file.exists():
            import json
            with open(best_file) as f:
                return float(json.load(f).get("opt_threshold", 0.5))
        return 0.5

    # ------------------------------------------------------------------
    def _validate_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the list of required columns missing from the user file."""
        return [c for c in REQUIRED_COLS if c not in df.columns]

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tolerate header variants users may upload. Two passes:
          1. friendly display names (schema.py)  -> technical names
          2. canonical names (case-insensitive, stripped)
        So 'taxi_TIME_since_first' would still be a non-match, but
        'Seconds_since_First_transaction', 'v14' etc. all work.
        """
        from schema import technical
        df = technical(df)                       # display/variant -> V1..V28

        canonical = {c.lower(): c for c in REQUIRED_COLS}
        rename = {}
        for col in df.columns:
            key = str(col).strip().lower()
            if key in canonical and col != canonical[key]:
                rename[col] = canonical[key]
        return df.rename(columns=rename) if rename else df

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_labels(y_true) -> pd.Series:
        """
        Convert a user-supplied ground-truth column into binary 0/1 integers:
          * numerics → astype(int)
          * common text labels ('Fraud', 'Legitimate', 'Yes', 'No', ...) → 0/1
        Raises ValueError when labels cannot be interpreted as classes.
        """
        ser = y_true.copy()
        if pd.api.types.is_bool_dtype(ser):
            return ser.astype(int)
        if pd.api.types.is_numeric_dtype(ser):
            return ser.astype(int)

        mapping = {"fraud": 1, "fraudulent": 1, "1": 1, "true": 1, "yes": 1,
                   "legitimate": 0, "normal": 0, "legit": 0, "0": 0,
                   "false": 0, "no": 0}
        lower = ser.astype(str).str.strip().str.lower()
        mapped = lower.map(mapping)
        if mapped.isna().any():
            raise ValueError(
                "Cannot parse the 'Class' column — expected 0/1 or text labels "
                "like 'Fraud'/'Legitimate'."
            )
        return mapped.astype(int)

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame,
                threshold: Optional[float] = None) -> PredictionResult:
        """Score a batch of new transactions and return labels + metrics.

        `threshold` overrides the default F1-optimal threshold for this call
        (used by UIs with a threshold control).
        """
        threshold = self.default_threshold if threshold is None else float(threshold)
        df = self.normalize_columns(df)
        if len(df) == 0:
            raise ValueError("The uploaded file contains no rows.")

        missing = self._validate_columns(df)
        if missing:
            raise ValueError(
                "Missing required columns: " + ", ".join(missing) +
                "\nExpected: Time, V1..V28, Amount"
            )

        # --- 1. replicate the training feature engineering -----------------
        features = df.loc[:, cfg.V_COLS].copy()
        scaled = self.scaler.transform(df[[cfg.AMOUNT_COL, cfg.TIME_COL]])
        features[cfg.TIME_SCALED] = scaled[:, 1]       # [Amount, Time] order
        features[cfg.AMOUNT_SCALED] = scaled[:, 0]
        features = features[cfg.FEATURE_COLS]          # enforce exact column order

        # --- 2. probability of fraud ---------------------------------------
        proba = self.model.predict_proba(features)[:, 1]

        # --- 3. threshold-based labels --------------------------------------
        out = df.copy()
        out["Fraud_Probability"] = proba
        out["Prediction"] = np.where(proba >= threshold,
                                     "Fraud", "Legitimate")

        # --- 4. optional evaluation when ground truth is provided -----------
        metrics = None
        if GROUND_TRUTH_COL in out.columns:
            try:
                y_true = self._coerce_labels(out[GROUND_TRUTH_COL])
                metrics = self._evaluate(y_true, proba, threshold)
            except Exception as exc:           # noqa: BLE001 — degrade gracefully
                metrics = {"error": str(exc), "single_class_batch": False}

        flagged = out[out["Prediction"] == "Fraud"].copy()
        return PredictionResult(
            table=out,
            fraud_count=int((out["Prediction"] == "Fraud").sum()),
            legit_count=int((out["Prediction"] == "Legitimate").sum()),
            metrics=metrics,
            flagged=flagged,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(y_true, proba, threshold: float = 0.5) -> dict:
        """Computes threshold-free + thresholded metrics vs ground truth.

        Handles batches that contain only ONE class (common with small or
        very unbalanced samples) by degrading gracefully to None for the
        class-sensitive metrics.
        """
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        pred = (proba >= threshold).astype(int)
        y_true = y_true.astype(int)

        n_classes = int(np.unique(y_true).size)
        pr_auc = roc_auc = None
        if n_classes == 2:
            pr_auc = round(float(average_precision_score(y_true, proba)), 4)
            roc_auc = round(float(roc_auc_score(y_true, proba)), 4)

        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        if tp + fp == 0:
            precision = None
        else:
            precision = round(float(tp / (tp + fp)), 4)
        recall = round(float(tp / (tp + fn)), 4) if (tp + fn) else None
        f1 = round(float(f1_score(y_true, pred, zero_division=0)), 4)

        return {
            "PR_AUC": pr_auc,
            "ROC_AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "single_class_batch": n_classes != 2,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def format_probability(p: float, decimals: int = 4) -> str:
        """Human-friendly percent string, e.g. 0.8732 → '87.32%'."""
        return f"{100 * p:.{decimals}f}%"

    # ------------------------------------------------------------------
    def load_raw_sample(self, n: int = 5, with_ground_truth: bool = True,
                        fraud_share: float = 0.4) -> pd.DataFrame:
        """
        Demo helper: returns REAL held-out test rows (RAW Amount/Time after
        reversing the scaling) so the app can showcase classification on data
        that looks exactly like a user upload. Enforces a mix that INCLUDES
        fraud rows so detection is visibly demonstrated.
        """
        X_test = pd.read_pickle(cfg.DATA_DIR / "X_test.pkl")
        y_test = pd.read_pickle(cfg.DATA_DIR / "y_test.pkl").to_numpy()

        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        n_fraud = min(int(round(n * fraud_share)), len(fraud_idx))
        n_legit = n - n_fraud
        sample_idx = np.concatenate([fraud_idx[:n_fraud], legit_idx[:n_legit]])
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        rng.shuffle(sample_idx)

        sample = X_test.iloc[sample_idx].copy()
        scaled = sample[[cfg.AMOUNT_SCALED, cfg.TIME_SCALED]].to_numpy()
        raw_vals = self.scaler.inverse_transform(scaled)     # [Amount, Time]
        sample[cfg.AMOUNT_COL] = raw_vals[:, 0]
        sample[cfg.TIME_COL] = raw_vals[:, 1]
        sample = sample[REQUIRED_COLS]                       # raw column order
        if with_ground_truth:
            sample[GROUND_TRUTH_COL] = y_test[sample_idx]
        return sample


# ---------------------------------------------------------------------------
# Standalone sanity check: score a small slice of the real test set.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = FraudPredictor()
    X_test = pd.read_pickle(cfg.DATA_DIR / "X_test.pkl")
    y_test = pd.read_pickle(cfg.DATA_DIR / "y_test.pkl")
    raw_test = X_test.copy()
    raw_test[cfg.AMOUNT_COL] = np.nan   # placeholders: real values come from
    raw_test[cfg.TIME_COL] = np.nan     # the unscaled test frame below

    # Reconstruct the raw test frame (reverse the scaling for a true end-to-end test)
    scaler = predictor.scaler
    amounts = scaler.inverse_transform(
        np.column_stack([X_test[cfg.AMOUNT_SCALED].to_numpy(),
                         X_test[cfg.TIME_SCALED].to_numpy()])
    )
    raw_test[cfg.AMOUNT_COL] = amounts[:, 0]
    raw_test[cfg.TIME_COL] = amounts[:, 1]
    raw_test[cfg.TARGET_COL] = y_test.to_numpy()

    result = predictor.predict(raw_test.head(5000))
    print(f"Sanity check on 5,000 raw test rows -> "
          f"{result.fraud_count} flagged fraud | {result.legit_count} legitimate")
    print("Metrics vs ground truth:", result.metrics)