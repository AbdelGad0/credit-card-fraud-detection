"""
config.py — Central configuration for the Credit Card Fraud Detection pipeline.

All scripts import paths, column names, and the global random seed from here so
the pipeline is reproducible and every stage uses identical settings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the project (parent of the config file's folder)
ROOT = Path(__file__).resolve().parent

# Raw dataset
DATA_RAW = ROOT / "creditcard.csv"

# Processed intermediate artifacts (splits, resampled data, scaler)
DATA_DIR = ROOT / "outputs" / "data"

# Trained model artifacts
MODEL_DIR = ROOT / "outputs" / "models"

# Figures (EDA, evaluation curves, SHAP plots)
FIG_DIR = ROOT / "outputs" / "figures"

# Tabular results (metrics comparison, EDA summary)
RESULTS_DIR = ROOT / "outputs"

# Ensure all output folders exist on import
for _dir in (DATA_DIR, MODEL_DIR, FIG_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------
TARGET_COL = "Class"
AMOUNT_COL = "Amount"
TIME_COL = "Time"

# Scaled versions of Time / Amount produced by the preprocessing stage
TIME_SCALED = "Time_Scaled"
AMOUNT_SCALED = "Amount_Scaled"

# Columns used as model features:
# V1..V28 (already PCA-transformed) + the scaled Time and Amount.
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = V_COLS + [TIME_SCALED, AMOUNT_SCALED]

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2