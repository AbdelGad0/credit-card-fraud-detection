"""
05_evaluation.py — Step 5: Model Evaluation (NO accuracy).

For every trained model variant, on the untouched 0.17%-fraud TEST set, report:
  * Precision, Recall, F1          (at 0.5 threshold AND at an F1-optimal threshold)
  * Confusion Matrix (TP/FP/TN/FN)
  * PR-AUC  (average precision)
  * ROC-AUC

Then produce the PR/ROC curves, a comparison CSV, and select the best model
(highest PR-AUC) for interpretation in Step 6.

Side effects
------------
- outputs/metrics_comparison.csv
- outputs/best_model.json
- outputs/figures/eval_precision_recall_curves.png
- outputs/figures/eval_roc_curves.png
- outputs/figures/eval_confusion_matrices.png
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config as cfg

# ---------------------------------------------------------------------------
# 1. Load test set + every trained model
# ---------------------------------------------------------------------------
X_test = pd.read_pickle(cfg.DATA_DIR / "X_test.pkl")
y_test = pd.read_pickle(cfg.DATA_DIR / "y_test.pkl")

with open(cfg.MODEL_DIR / "models_metadata.json") as f:
    meta = json.load(f)

variants = meta["model_variants"]          # [{model, variant, path}, ...]


def best_f1_threshold(y_true, proba):
    """Threshold on the PR curve that maximizes F1 (operational decision)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1_scores = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    idx = int(np.argmax(f1_scores[:-1]))   # PR curve returns one extra point
    return thr[idx], f1_scores[idx]


results = []
fig_pr, ax_pr = plt.subplots(figsize=(9, 7))
fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
cm_figs = {}

for v in variants:
    name, variant = v["model"], v["variant"]
    label = f"{name} ({variant})"
    model = joblib.load(cfg.MODEL_DIR / v["path"])
    proba = model.predict_proba(X_test)[:, 1]

    # ----- Core thresholds-free metrics ------------------------------------
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    # ----- Metrics at default threshold 0.5 ---------------------------------
    pred_default = (proba >= 0.5).astype(int)
    p05 = precision_score(y_test, pred_default)
    r05 = recall_score(y_test, pred_default)
    f05 = f1_score(y_test, pred_default)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_default).ravel()

    # ----- Metrics at F1-optimal threshold ----------------------------------
    thr_opt, _ = best_f1_threshold(y_test, proba)
    pred_opt = (proba >= thr_opt).astype(int)
    popt = precision_score(y_test, pred_opt)
    ropt = recall_score(y_test, pred_opt)
    fopt = f1_score(y_test, pred_opt)

    results.append({
        "model": name, "variant": variant,
        "PR_AUC": round(pr_auc, 4), "ROC_AUC": round(roc_auc, 4),
        "Precision@0.5": round(p05, 4), "Recall@0.5": round(r05, 4), "F1@0.5": round(f05, 4),
        "TP@0.5": tp, "FP@0.5": fp, "FN@0.5": fn, "TN@0.5": tn,
        "opt_threshold": round(float(thr_opt), 4),
        "Precision@opt": round(popt, 4), "Recall@opt": round(ropt, 4), "F1@opt": round(fopt, 4),
    })

    # ----- Curves -----------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax_pr.plot(rec, prec, lw=1.5, label=f"{label} (PR-AUC={pr_auc:.3f})")
    ax_roc.plot(fpr, tpr, lw=1.5, label=f"{label} (ROC-AUC={roc_auc:.3f})")

    # Confusion matrix figure (all variants, 2x2 grid of models)
    cm_figs.setdefault(name, []).append((variant, confusion_matrix(y_test, pred_opt)))

    print(f"{label:45s} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
          f"F1@0.5={f05:.4f}  P@opt={popt:.4f} R@opt={ropt:.4f} F1@opt={fopt:.4f}")

# ---------------------------------------------------------------------------
# 2. Figures
# ---------------------------------------------------------------------------
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curves (held-out TEST set, 0.17% fraud)")
ax_pr.legend(loc="lower left", fontsize=8); ax_pr.grid(alpha=0.3)
fig_pr.tight_layout(); fig_pr.savefig(cfg.FIG_DIR / "eval_precision_recall_curves.png", dpi=150)

ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves (held-out TEST set)")
ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
ax_roc.legend(loc="lower right", fontsize=8); ax_roc.grid(alpha=0.3)
fig_roc.tight_layout(); fig_roc.savefig(cfg.FIG_DIR / "eval_roc_curves.png", dpi=150)

# Confusion matrices: one 2x2 heatmap per algorithm (both variants)
n_models = len(cm_figs)
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
axes_cm = axes_cm.ravel()
for i, (model_name, cm_list) in enumerate(cm_figs.items()):
    for variant, cm_ in cm_list:
        inner = f"{model_name}_{variant}"[0].upper() if False else model_name
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(cm_, display_labels=["Legit", "Fraud"])
        disp.plot(ax=axes_cm[i], colorbar=False, cmap="Blues", values_format="d")
        axes_cm[i].set_title(f"{model_name} @ F1-optimal threshold")
for j in range(len(cm_figs), len(axes_cm)):
    axes_cm[j].axis("off")
fig_cm.suptitle("Confusion Matrices on held-out TEST set (threshold tuned for F1)")
fig_cm.tight_layout(); fig_cm.savefig(cfg.FIG_DIR / "eval_confusion_matrices.png", dpi=150)
plt.close("all")

# ---------------------------------------------------------------------------
# 3. Persist comparison + select best model by PR-AUC
# ---------------------------------------------------------------------------
df_results = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
df_results.to_csv(cfg.RESULTS_DIR / "metrics_comparison.csv", index=False)

best_row = df_results.iloc[0]
with open(cfg.RESULTS_DIR / "best_model.json", "w") as f:
    json.dump({
        "model": best_row["model"],
        "variant": best_row["variant"],
        "PR_AUC": float(best_row["PR_AUC"]),
        "ROC_AUC": float(best_row["ROC_AUC"]),
        "opt_threshold": float(best_row["opt_threshold"]),
        "file": f'{best_row["model"]}_{best_row["variant"]}.pkl',
    }, f, indent=2)

print("\n=== BEST MODEL BY PR-AUC ===")
print(best_row.to_string())
print(f"\nSaved: metrics_comparison.csv, best_model.json, eval_*.png")
print("Evaluation complete.")