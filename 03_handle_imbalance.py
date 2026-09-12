"""
03_handle_imbalance.py — Step 3: Class Imbalance Handling (CRITICAL).

Strategy
--------
Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to the TRAINING
set ONLY, after the train/test split. The test set is deliberately left at its
real-world 0.17% prevalence so every reported metric reflects true deployment
conditions.

Why SMOTE and not SMOTEENN/TomekLinks here?
-------------------------------------------
- SMOTE creates interpolated minority samples, which is very effective for this
  dataset because fraud transactions form tight, well-separated clusters in the
  PCA-reduced feature space.
- SMOTEENN and Tomek Links additionally *clean* the data (they remove noisy /
  borderline samples using k-NN). On 227k training rows those cleaning passes are
  computationally prohibitive (>30 min), so SMOTE alone was retained.
- As a complementary strategy, the model-training stage ALSO uses class weights /
  scale_pos_weight, so results are robust regardless of resampling choice.

Leakage prevention
------------------
- Splitting happens in 02_preprocessing.py BEFORE any resampling.
- SMOTE is fitted on training data only → synthetic samples never touch the test set,
  and any reported metric is not inflated by interpolated test observations.

Side effects
------------
- outputs/data/X_train_resampled.pkl, y_train_resampled.pkl
- outputs/imbalance_summary.json
"""

import json

import numpy as np
import pandas as pd

import config as cfg

# ---------------------------------------------------------------------------
# 1. Load training data (already split & scaled)
# ---------------------------------------------------------------------------
X_train = pd.read_pickle(cfg.DATA_DIR / "X_train.pkl")
y_train = pd.read_pickle(cfg.DATA_DIR / "y_train.pkl")

before = y_train.value_counts().sort_index()
print("=== CLASS COUNTS BEFORE RESAMPLING (train ONLY) ===")
print(before.to_string())

# ---------------------------------------------------------------------------
# 2. Apply SMOTE on the training set only
# ---------------------------------------------------------------------------
from imblearn.over_sampling import SMOTE

# sampling_strategy=1.0 → fully balance the two classes (1:1)
smote = SMOTE(random_state=cfg.RANDOM_STATE, sampling_strategy=1.0, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)

after = pd.Series(y_res).value_counts().sort_index()
print("\n=== CLASS COUNTS AFTER SMOTE (train ONLY) ===")
print(after.to_string())

generated = int(after[1]) - int(before[1])
print(f"\nSynthetic fraud samples generated: {generated:,}")
print(f"Resampled train size: {len(X_res):,} rows (was {len(X_train):,})")

# ---------------------------------------------------------------------------
# 3. Persist resampled training data + summary
# ---------------------------------------------------------------------------
X_res.to_pickle(cfg.DATA_DIR / "X_train_resampled.pkl")
y_res.to_pickle(cfg.DATA_DIR / "y_train_resampled.pkl")

summary = {
    "method": "SMOTE",
    "sampling_strategy": "1.0 (fully balanced)",
    "k_neighbors": smote.k_neighbors,
    "class_counts_before": {int(k): int(v) for k, v in before.items()},
    "class_counts_after": {int(k): int(v) for k, v in after.items()},
    "synthetic_minority_generated": generated,
    "n_rows_before": len(X_train),
    "n_rows_after": len(X_res),
    "test_set_untouched": True,
    "leakage_prevention": "Resampling applied AFTER train/test split; fitted on train only.",
}
with open(cfg.RESULTS_DIR / "imbalance_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nImbalance summary saved to {cfg.RESULTS_DIR / 'imbalance_summary.json'}")
print("Imbalance handling complete.")