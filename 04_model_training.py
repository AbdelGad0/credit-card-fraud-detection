"""
04_model_training.py — Step 4: Train & Compare Multiple Classifiers.

Models
------
1. LogisticRegression          (linear baseline)
2. RandomForestClassifier      (ensemble / bagging)
3. XGBClassifier               (gradient boosting)
4. LGBMClassifier              (gradient boosting)

Each algorithm is trained in TWO imbalance-handling variants:
  A) "weighted"  → class_weight='balanced' / scale_pos_weight on ORIGINAL train
  B) "smote"     → no class weights, trained on the SMOTE-balanced train set
This produces 8 fitted models for a thorough, honest comparison in Step 5.

Hyperparameters are selected with small cross-validated searches on the
ORIGINAL training data (cv=3, scoring='average_precision' i.e. PR-AUC), then
the winning parameters are re-applied to both variants.

Side effects
------------
- outputs/models/{model_name}_{weighted|smote}.pkl
- outputs/models/models_metadata.json   (params, cv scores, fit time)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier

import config as cfg

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
X_og = pd.read_pickle(cfg.DATA_DIR / "X_train.pkl")            # original train
y_og = pd.read_pickle(cfg.DATA_DIR / "y_train.pkl")
X_res = pd.read_pickle(cfg.DATA_DIR / "X_train_resampled.pkl")  # SMOTE-balanced train
y_res = pd.read_pickle(cfg.DATA_DIR / "y_train_resampled.pkl")

# scale_pos_weight for original train ≈ majority / minority
neg, pos = y_og.value_counts().sort_index().values
SCALE_POS_WEIGHT = float(neg / pos)
print(f"scale_pos_weight (neg/pos) = {SCALE_POS_WEIGHT:.1f}")


# ---------------------------------------------------------------------------
# 1. Model zoo — (estimator skeleton, search space, search type)
# ---------------------------------------------------------------------------
# Solver 'liblinear' supports class_weight and is fast on 30 features.
lr = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced",
                        random_state=cfg.RANDOM_STATE)
rf = RandomForestClassifier(n_jobs=-1, random_state=cfg.RANDOM_STATE,
                            class_weight="balanced")
xgb = XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1,
                    random_state=cfg.RANDOM_STATE)
lgbm = LGBMClassifier(verbosity=-1, n_jobs=-1, random_state=cfg.RANDOM_STATE)

MODEL_ZOO = {
    "LogisticRegression": {
        "estimator": lr,
        "search": GridSearchCV,
        "space": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "RandomForest": {
        "estimator": rf,
        "search": RandomizedSearchCV,
        "space": {"n_estimators": [150, 300],
                  "max_depth": [10, 20, None],
                  "min_samples_leaf": [1, 5]},
        "n_iter": 4,
    },
    "XGBoost": {
        "estimator": xgb,
        "search": RandomizedSearchCV,
        "space": {"n_estimators": [200, 350],
                  "max_depth": [4, 6],
                  "learning_rate": [0.05, 0.1],
                  "scale_pos_weight": [SCALE_POS_WEIGHT]},
        "n_iter": 4,
    },
    "LightGBM": {
        "estimator": lgbm,
        "search": RandomizedSearchCV,
        "space": {"n_estimators": [250, 400],
                  "num_leaves": [31, 63],
                  "learning_rate": [0.05, 0.1],
                  "scale_pos_weight": [SCALE_POS_WEIGHT]},
        "n_iter": 4,
    },
}


def select_best_params(name: str, spec: dict, X: pd.DataFrame, y: pd.Series):
    """Run a small CV search on the ORIGINAL train data and return best params."""
    search_cls = spec["search"]
    kwargs = {
        "estimator": spec["estimator"],
        "param_grid" if search_cls is GridSearchCV else "param_distributions": spec["space"],
        "cv": 3,
        "scoring": "average_precision",   # optimize PR-AUC (relevant for rare fraud)
        "n_jobs": 1,                      # estimators parallelize internally
        "refit": True,
    }
    if search_cls is RandomizedSearchCV:
        kwargs["n_iter"] = spec["n_iter"]
        kwargs["random_state"] = cfg.RANDOM_STATE
    search = search_cls(**kwargs)
    search.fit(X, y)
    return (
        search.best_params_,
        search.best_score_,           # mean CV PR-AUC on original train
        search.cv_results_["mean_test_score"].max(),
        search.best_estimator_,       # already refitted on full original train
    )


# ---------------------------------------------------------------------------
# 2. Train both variants of every model
# ---------------------------------------------------------------------------
artifacts = []
metadata = {}

for name, spec in MODEL_ZOO.items():
    print(f"\n=== {name}: tuning (cv=3) on ORIGINAL train ===")
    best_params, cv_pr_auc, _, best_estimator = select_best_params(name, spec, X_og, y_og)
    print(f"  best params: {best_params}")
    print(f"  mean CV PR-AUC (original): {cv_pr_auc:.5f}")

    # --- Variant A: weighted, trained on ORIGINAL data ----------------------
    t0 = time.time()
    model_weighted = best_estimator            # already refit on original train
    fit_time_weighted = round(time.time() - t0, 1)
    joblib.dump(model_weighted, cfg.MODEL_DIR / f"{name}_weighted.pkl")
    print(f"  [{name}_weighted] fit time {fit_time_weighted}s")

    # --- Variant B: SMOTE, trained on balanced data (neutralize weights) ----
    neutral = dict(best_params)
    if "scale_pos_weight" in neutral:
        neutral["scale_pos_weight"] = 1.0      # data already balanced
    if name == "LogisticRegression":
        neutral.pop("class_weight", None)
    if name in ("RandomForest",):
        neutral.pop("class_weight", None)
    if name in ("XGBoost", "LightGBM"):
        neutral.pop("scale_pos_weight", None)

    t0 = time.time()
    model_smote = spec["estimator"].set_params(**neutral) if neutral else spec["estimator"]
    # For LR/RF the estimator skeleton already carries class_weight='balanced';
    # clear it so the balanced SMOTE data is not double-weighted.
    if hasattr(model_smote, "class_weight") and model_smote.class_weight is not None:
        model_smote.set_params(class_weight=None)
    model_smote.fit(X_res, y_res)
    fit_time_smote = round(time.time() - t0, 1)
    joblib.dump(model_smote, cfg.MODEL_DIR / f"{name}_smote.pkl")
    print(f"  [{name}_smote] fit time {fit_time_smote}s")

    artifacts.append({"model": name, "variant": "weighted", "path": f"{name}_weighted.pkl",
                      "time_s": fit_time_weighted})
    artifacts.append({"model": name, "variant": "smote", "path": f"{name}_smote.pkl",
                      "time_s": fit_time_smote})
    metadata[name] = {
        "best_params_original": {k: (float(v) if isinstance(v, np.floating) else v)
                                 for k, v in best_params.items()},
        "mean_cv_PR_AUC_original": round(float(cv_pr_auc), 5),
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }

# ---------------------------------------------------------------------------
# 3. Persist metadata
# ---------------------------------------------------------------------------
with open(cfg.MODEL_DIR / "models_metadata.json", "w") as f:
    json.dump({"model_variants": artifacts, "hyperparameters": metadata}, f, indent=2)

print(f"\nSaved {len(artifacts)} model artifacts to {cfg.MODEL_DIR}")
print("Model training complete.")