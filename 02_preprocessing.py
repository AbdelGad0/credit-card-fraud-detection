"""
02_preprocessing.py — Step 2: Scaling & Stratified Train/Test Split.

Tasks
-----
1. Scale `Time` and `Amount` with RobustScaler (robust to the heavy right tail
   of transaction amounts shown in EDA).
2. Split the data into training / testing sets with STRATIFIED sampling
   (test_size=0.2, random_state=42) so the ~0.17% fraud ratio is preserved in
   both folds.
3. The scaler is fitted on the TRAINING portion only, then applied to the test
   set — this prevents information from the test set leaking into training.

Side effects
------------
- outputs/data/X_train.pkl, y_train.pkl
- outputs/data/X_test.pkl,  y_test.pkl
- outputs/data/scaler.pkl          (fitted RobustScaler)
- outputs/preprocessing_summary.json
"""

import json

import joblib
import pandas as pd

import config as cfg

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
df = pd.read_csv(cfg.DATA_RAW)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

# ---------------------------------------------------------------------------
# 2. Split BEFORE scaling/resampling (leakage-safe order of operations)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

y = df[cfg.TARGET_COL].astype(int)          # target: 1 = fraud, 0 = legit
X = df.drop(columns=[cfg.TARGET_COL])       # raw predictors (Time, V1..V28, Amount)

# Stratified split preserves the (already tiny) fraud ratio in both folds.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg.TEST_SIZE,
    random_state=cfg.RANDOM_STATE,
    stratify=y,          # keeps ~0.17% fraud in BOTH train and test
    shuffle=True,
)

train_fraud_pct = 100.0 * y_train.sum() / len(y_train)
test_fraud_pct = 100.0 * y_test.sum() / len(y_test)
print(f"Train: {len(X_train):,} rows (fraud {train_fraud_pct:.3f}%)")
print(f"Test:  {len(X_test):,} rows (fraud {test_fraud_pct:.3f}%)")

# ---------------------------------------------------------------------------
# 3. Scale Time & Amount with RobustScaler (fit ONLY on train)
# ---------------------------------------------------------------------------
from sklearn.preprocessing import RobustScaler

# RobustScaler uses median + IQR → unaffected by the huge Amount outliers.
scaler = RobustScaler()
X_train[[cfg.AMOUNT_SCALED, cfg.TIME_SCALED]] = scaler.fit_transform(
    X_train[[cfg.AMOUNT_COL, cfg.TIME_COL]]
)
# Apply the SAME fitted scaler to the test set (no refit → no leakage).
X_test[[cfg.AMOUNT_SCALED, cfg.TIME_SCALED]] = scaler.transform(
    X_test[[cfg.AMOUNT_COL, cfg.TIME_COL]]
)

# Drop the unscaled originals; keep only model features.
X_train = X_train[cfg.FEATURE_COLS]
X_test = X_test[cfg.FEATURE_COLS]
print("\nFinal feature set:", cfg.FEATURE_COLS)

# ---------------------------------------------------------------------------
# 4. Persist artifacts
# ---------------------------------------------------------------------------
X_train.to_pickle(cfg.DATA_DIR / "X_train.pkl")
y_train.to_pickle(cfg.DATA_DIR / "y_train.pkl")
X_test.to_pickle(cfg.DATA_DIR / "X_test.pkl")
y_test.to_pickle(cfg.DATA_DIR / "y_test.pkl")
joblib.dump(scaler, cfg.DATA_DIR / "scaler.pkl")

summary = {
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "train_fraud_pct": round(train_fraud_pct, 4),
    "test_fraud_pct": round(test_fraud_pct, 4),
    "scaler": "RobustScaler",
    "scaled_columns": [cfg.AMOUNT_SCALED, cfg.TIME_SCALED],
    "n_features": X_train.shape[1],
    "medians": scaler.center_.tolist(),
    "iqr_scales": scaler.scale_.tolist(),
}
with open(cfg.RESULTS_DIR / "preprocessing_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nArtifacts saved under {cfg.DATA_DIR}")
print("Preprocessing complete.")