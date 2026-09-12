"""
06_interpretation.py — Step 6: Model Interpretation / Explainable AI (SHAP).

Explains the BEST model from Step 5 (LightGBM-smote, PR-AUC = 0.880) on the
held-out TEST set using SHAP:
  1. SHAP beeswarm plot     — direction & magnitude of each feature's influence
  2. SHAP mean-|SHAP| bar   — global feature importance (model-agnostic)
  3. Native LightGBM gain importances — cross-check with the boosted trees
  4. SHAP dependence plot   — nonlinear relationship for the top feature

SHAP background: a stratified sample of the test set that includes ALL frauds
(so fraud explanations are well-represented) plus random legit samples.

Side effects
------------
- outputs/figures/shap_beeswarm.png
- outputs/figures/shap_bar.png
- outputs/figures/feature_importance_gain.png
- outputs/figures/shap_dependence_top1.png
- outputs/shap_top_features.json
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config as cfg

# ---------------------------------------------------------------------------
# 1. Load best model + data
# ---------------------------------------------------------------------------
with open(cfg.RESULTS_DIR / "best_model.json") as f:
    best = json.load(f)
print(f"Best model: {best['model']} ({best['variant']}) — PR-AUC {best['PR_AUC']}")

model = joblib.load(cfg.MODEL_DIR / best["file"])
X_test = pd.read_pickle(cfg.DATA_DIR / "X_test.pkl")
y_test = pd.read_pickle(cfg.DATA_DIR / "y_test.pkl")

# ---------------------------------------------------------------------------
# 2. Stratified explanation sample (all frauds + balanced legit)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(cfg.RANDOM_STATE)
fraud_idx = np.where(y_test == 1)[0]
n_legit = min(len(fraud_idx) * 20, (y_test == 0).sum())   # 20 legit per fraud max
legit_idx = rng.choice(np.where(y_test == 0)[0], 2000, replace=False)
sample_idx = np.concatenate([fraud_idx, legit_idx])
rng.shuffle(sample_idx)

X_sample = X_test.iloc[sample_idx]
print(f"Explanation sample size: {len(X_sample)} (contains all {len(fraud_idx)} test frauds)")

# ---------------------------------------------------------------------------
# 3. SHAP values (TreeExplainer for LightGBM)
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)                    # model-agnostic wrapper
shap_values = explainer.shap_values(X_sample)            # shape (n_sample, 30)
print(f"SHAP values shape: {shap_values.shape}")

# Global summary
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1]
top_features = [X_sample.columns[i] for i in top_idx]

# --- Beeswarm (SHAP summary) plot ------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
plt.title("SHAP impact on model output (LightGBM-smote, test set)")
plt.tight_layout()
plt.savefig(cfg.FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Mean |SHAP| bar plot ----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(X_sample.columns[top_idx][:15][::-1], mean_abs_shap[top_idx][:15][::-1], color="#4C72B0")
ax.set_xlabel("mean |SHAP value| (average impact on model output)")
ax.set_title("Global feature importance (SHAP)")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(cfg.FIG_DIR / "shap_bar.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Native LightGBM gain importances (cross-check)
# ---------------------------------------------------------------------------
lgb_imp = model.booster_.feature_importance(importance_type="gain")
lgb_imp_df = pd.DataFrame({"feature": X_sample.columns, "gain": lgb_imp}).sort_values("gain", ascending=False)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(lgb_imp_df["feature"][:15][::-1], lgb_imp_df["gain"][:15][::-1], color="#C44E52")
ax.set_xlabel("Gain (total information gain from splits on feature)")
ax.set_title("LightGBM native feature importance (gain)")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(cfg.FIG_DIR / "feature_importance_gain.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Dependence plot for the #1 SHAP feature
# ---------------------------------------------------------------------------
top1 = top_features[0]
fig, ax = plt.subplots(figsize=(9, 7))
shap.dependence_plot(top1, shap_values, X_sample, ax=ax, show=False)
plt.title(f"SHAP dependence: {top1}")
plt.tight_layout()
plt.savefig(cfg.FIG_DIR / "shap_dependence_top1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Persist top-feature summary
# ---------------------------------------------------------------------------
shap_top = {f: round(float(mean_abs_shap[i]), 6) for f, i in
            zip(top_features, top_idx)}
with open(cfg.RESULTS_DIR / "shap_top_features.json", "w") as f:
    json.dump({"top_features_shap": shap_top,
               "top_features_gain": lgb_imp_df["feature"].head(15).tolist()}, f, indent=2)

print("\n=== TOP 10 FEATURES (mean |SHAP|) ===")
for f in top_features[:10]:
    print(f"  {f}: {mean_abs_shap[list(X_sample.columns).index(f)]:.5f}")
print("\nFigures + shap_top_features.json saved.")
print("Interpretation complete.")