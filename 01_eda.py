"""
01_eda.py — Step 1: Data Loading & Exploratory Data Analysis (EDA).

Tasks
-----
1. Load the raw `creditcard.csv` dataset.
2. Inspect shape, data types and missing values.
3. Explicitly quantify the class imbalance ratio.
4. Explore Time / Amount distributions and feature-class correlations.
5. Export figures + a machine-readable summary for `project.md`.

Side effects
------------
- outputs/figures/eda_class_distribution.png
- outputs/figures/eda_amount_distribution.png
- outputs/figures/eda_time_distribution.png
- outputs/figures/eda_correlation_heatmap.png
- outputs/eda_summary.json
"""

import json

import matplotlib

matplotlib.use("Agg")  # headless backend: no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(cfg.DATA_RAW)
print("Dataset shape (rows, columns):", df.shape)


# ---------------------------------------------------------------------------
# 2. Inspect structure
# ---------------------------------------------------------------------------
def inspect_structure(df: pd.DataFrame) -> dict:
    """Return basic structural facts about the DataFrame."""
    n_rows, n_cols = df.shape
    missing = int(df.isna().sum().sum())
    missing_per_feature = df.isna().sum()
    dtypes = {col: str(t) for col, t in df.dtypes.items()}
    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_missing_cells": missing,
        "missing_per_feature": missing_per_feature[missing_per_feature > 0].to_dict(),
        "dtypes": dtypes,
    }


structure = inspect_structure(df)
print(f"Total missing cells: {structure['n_missing_cells']}")
assert structure["n_missing_cells"] == 0, "Unexpected missing values found!"


# ---------------------------------------------------------------------------
# 3. Class imbalance analysis
# ---------------------------------------------------------------------------
def class_imbalance_report(df: pd.DataFrame) -> dict:
    """Quantify the positive-class ('fraud') imbalance ratio."""
    counts = df[cfg.TARGET_COL].value_counts().sort_index()
    n_fraud = int(counts.get(1, 0))
    n_legit = int(counts.get(0, 0))
    total = n_fraud + n_legit
    fraud_pct = 100.0 * n_fraud / total
    return {
        "legit_count": n_legit,
        "fraud_count": n_fraud,
        "total": int(total),
        "fraud_percentage": round(fraud_pct, 4),
        "imbalance_ratio": f"1 : {total / n_fraud:.0f}",
    }


imbalance = class_imbalance_report(df)
print("\n=== CLASS IMBALANCE ===")
print(json.dumps(imbalance, indent=2))

# --- Figure: class distribution -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x=cfg.TARGET_COL, hue=cfg.TARGET_COL,
              palette=["#4C72B0", "#C44E52"], legend=False, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
for i, p in enumerate(ax.patches):
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
ax.set_title(f"Class Distribution — Fraud only {imbalance['fraud_percentage']:.3f}% of data")
ax.set_ylabel("Number of transactions")
fig.tight_layout()
fig.savefig(cfg.FIG_DIR / "eda_class_distribution.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Feature distributions
# ---------------------------------------------------------------------------
# --- Amount ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(x=df.loc[df[cfg.TARGET_COL] == 1, cfg.AMOUNT_COL], bins=60, color="#C44E52", ax=axes[0])
axes[0].set_title("Fraud — Transaction Amount (raw)")
axes[0].set_xlabel("Amount (EUR)")
sns.histplot(x=df.loc[df[cfg.TARGET_COL] == 0, cfg.AMOUNT_COL], bins=60, color="#4C72B0", ax=axes[1])
axes[1].set_title("Legitimate — Transaction Amount (raw)")
axes[1].set_xlabel("Amount (EUR)")
fig.tight_layout()
fig.savefig(cfg.FIG_DIR / "eda_amount_distribution.png", dpi=150)
plt.close(fig)

# --- Time -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(x=df.loc[df[cfg.TARGET_COL] == 1, cfg.TIME_COL], bins=60, color="#C44E52", ax=axes[0])
axes[0].set_title("Fraud — Time of transaction (seconds after first)")
sns.histplot(x=df.loc[df[cfg.TARGET_COL] == 0, cfg.TIME_COL], bins=60, color="#4C72B0", ax=axes[1])
axes[1].set_title("Legitimate — Time of transaction")
fig.tight_layout()
fig.savefig(cfg.FIG_DIR / "eda_time_distribution.png", dpi=150)
plt.close(fig)

# Descriptive stats per class
amount_stats = df.groupby(cfg.TARGET_COL)[cfg.AMOUNT_COL].describe()[["mean", "50%", "75%", "max"]]
print("\n=== AMOUNT STATS BY CLASS ===")
print(amount_stats.round(2).to_string())


# ---------------------------------------------------------------------------
# 5. Correlation with the target
# ---------------------------------------------------------------------------
# At this stage only the RAW Time/Amount exist (scaled versions come later).
eda_feature_cols = cfg.V_COLS + [cfg.TIME_COL, cfg.AMOUNT_COL]
corr_with_target = df[eda_feature_cols + [cfg.TARGET_COL]].corr()[cfg.TARGET_COL].drop(cfg.TARGET_COL)
top_pos = corr_with_target.nlargest(5)
top_neg = corr_with_target.nsmallest(5)
print("\n=== TOP 5 POSITIVE CORRELATIONS WITH CLASS ===")
print(top_pos.round(4).to_string())
print("\n=== TOP 5 NEGATIVE CORRELATIONS WITH CLASS ===")
print(top_neg.round(4).to_string())

# --- Figure: correlation heatmap of strongest feature-to-target pairs -------
fig, ax = plt.subplots(figsize=(9, 8))
selected = list(top_pos.index) + list(top_neg.index)
corr_sub = df[selected + [cfg.TARGET_COL]].corr()
sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix (strongest features)")
fig.tight_layout()
fig.savefig(cfg.FIG_DIR / "eda_correlation_heatmap.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Persist summary for project.md
# ---------------------------------------------------------------------------
summary = {
    "structure": structure,
    "imbalance": imbalance,
    "amount_stats_by_class": amount_stats.round(2).to_dict(),
    "top_corr_with_target": {**top_pos.round(4).to_dict(), **top_neg.round(4).to_dict()},
}
with open(cfg.RESULTS_DIR / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nEDA summary saved to {cfg.RESULTS_DIR / 'eda_summary.json'}")
print("EDA complete.")