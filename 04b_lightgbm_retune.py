"""
04b_lightgbm_retune.py — Targeted re-tuning of LightGBM.

Why: in the initial pass LightGBM reached only 0.055 CV PR-AUC while XGBoost got
0.84. The default `min_child_samples=20` is far too high when each CV fold holds
only ~131 fraud samples (0.17% prevalence), so the learner cannot form pure
fraud leaves. We re-tune with much smaller leaf sizes and regularization knobs.

Produces (overwrites the earlier sub-par LightGBM variants):
- outputs/models/LightGBM_weighted.pkl
- outputs/models/LightGBM_smote.pkl
- updates outputs/models/models_metadata.json (LightGBM entry only)
"""

import json
import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

import config as cfg

X_og = pd.read_pickle(cfg.DATA_DIR / "X_train.pkl")
y_og = pd.read_pickle(cfg.DATA_DIR / "y_train.pkl")
X_res = pd.read_pickle(cfg.DATA_DIR / "X_train_resampled.pkl")
y_res = pd.read_pickle(cfg.DATA_DIR / "y_train_resampled.pkl")

neg, pos = y_og.value_counts().sort_index().values
SCALE_POS_WEIGHT = float(neg / pos)

est_base = LGBMClassifier(verbosity=-1, n_jobs=-1, random_state=cfg.RANDOM_STATE)
# Search space guided by a diagnostic that showed high-capacity + strongly
# regularized LightGBM (num_leaves=128, lr=0.03, min_child_samples=1) reaches
# ~0.85 PR-AUC on this dataset; the default (min_child_samples=20) fails badly.
space = {
    "n_estimators": [300, 500],
    "num_leaves": [64, 128, 200],
    "learning_rate": [0.03, 0.05, 0.08],
    "min_child_samples": [1, 5, 20],        # NEW: allow small fraud leaves
    "subsample": [0.6, 0.7, 0.8],
    "subsample_freq": [1],
    "colsample_bytree": [0.5, 0.7],
    "reg_lambda": [1.0, 2.0, 5.0],
    "scale_pos_weight": [SCALE_POS_WEIGHT],
}

print("=== LightGBM re-tuning (cv=3) on ORIGINAL train ===")
search = RandomizedSearchCV(
    est_base, space, n_iter=8, cv=3, scoring="average_precision",
    n_jobs=1, refit=True, random_state=cfg.RANDOM_STATE,
)
search.fit(X_og, y_og)
print(f"best params: {search.best_params_}")
print(f"mean CV PR-AUC (original): {search.best_score_:.5f}")

# Variant A — weighted on original train (best_estimator already refit on full train)
model_weighted = search.best_estimator_
joblib.dump(model_weighted, cfg.MODEL_DIR / "LightGBM_weighted.pkl")

# Variant B — SMOTE-balanced data, no class-weight skewing
neutral = {k: v for k, v in search.best_params_.items() if k != "scale_pos_weight"}
model_smote = est_base.set_params(**neutral)
t0 = time.time()
model_smote.fit(X_res, y_res)
print(f"[LightGBM_smote] fit time {time.time() - t0:.1f}s")
joblib.dump(model_smote, cfg.MODEL_DIR / "LightGBM_smote.pkl")

# Update metadata
with open(cfg.MODEL_DIR / "models_metadata.json") as f:
    metadata = json.load(f)
metadata["hyperparameters"]["LightGBM"] = {
    "best_params_original": {k: (float(v) if isinstance(v, (int, float)) else v)
                             for k, v in search.best_params_.items()},
    "mean_cv_PR_AUC_original": round(float(search.best_score_), 5),
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "note": "Re-tuned with min_child_samples <= 20; initial pass used LGBM default (20) -> 0.055 PR-AUC.",
}
with open(cfg.MODEL_DIR / "models_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("LightGBM re-tuning complete.")