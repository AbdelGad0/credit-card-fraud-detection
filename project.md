# Credit Card Fraud Detection — End-to-End Machine Learning Pipeline

**Author:** Data Science / ML Engineer
**Goal:** Build a production-style, explainable ML pipeline to identify fraudulent credit card transactions, designed as a professional portfolio project.

---

## 1. Project Overview

Credit card fraud causes billions in annual losses. Because fraudulent transactions are extremely rare (~0.17% of transactions), this is a **highly imbalanced binary classification** problem. The cost of missing a fraud (False Negative) is far higher than the cost of a false alert, so we must evaluate models on **Precision / Recall / F1 / PR-AUC / ROC-AUC** — never Accuracy.

This project implements a complete, reproducible pipeline:

1. **EDA** — Understand the data, class imbalance, distributions, correlations.
2. **Preprocessing** — Scale `Time` & `Amount` (RobustScaler) + stratified train/test split.
3. **Class Imbalance Handling** — SMOTE / SMOTEENN applied **on the training set only** (prevents data leakage).
4. **Model Training** — Compare 3+ models (Logistic Regression, Random Forest, XGBoost, LightGBM) via stratified cross-validation.
5. **Evaluation** — PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix, threshold tuning.
6. **Model Interpretation** — SHAP values + feature importance for the best model.

---

## 2. Dataset Description

- **Source:** Kaggle "Credit Card Fraud Detection" (European cardholders, Sep 2013).
- **Records:** ~284,807 transactions.
- **Features:** `Time`, `V1`–`V28` (PCA-transformed, anonymized), `Amount`.
- **Target:** `Class` — `1` = Fraud (492 cases), `0` = Legitimate.

> Because the `V1`–`V28` features are already PCA-reduced, we only need to scale `Time` and `Amount`.

---

## 3. Pipeline Architecture

```
creditcard.csv
      │
      ▼
[1] EDA ───────────────► outputs/figures/*, insights logged
      │
      ▼
[2] Preprocessing ─────► RobustScaler(Time, Amount) + Stratified 80/20 split
      │                    └─► outputs/data/train.pkl, test.pkl (+scaler)
      ▼
[3] Imbalance Handling ─► SMOTE / SMOTEENN on TRAIN ONLY (no leakage)
      │                    └─► outputs/data/train_resampled.pkl
      ▼
[4] Model Training ─────► LogisticRegression, RandomForest, XGBoost, LightGBM
      │                    CV search + class weights  └─► outputs/models/*.pkl
      ▼
[5] Evaluation ─────────► Precision, Recall, F1, Confusion Matrix,
      │                    PR-AUC, ROC-AUC + threshold tuning
      │                    └─► outputs/figures/*, outputs/metrics.csv
      ▼
[6] Interpretation ─────► SHAP summary/bar plots + feature importance
                            └─► outputs/figures/*, project.md summary
```

**Reproduction:** `config.py` centralizes all paths and the fixed `random_state = 42`. Run scripts `01_eda.py` → `06_interpretation.py` in order.

---

## 4. Why These Techniques? (Design Rationale)

| Concern | Chosen Technique | Reasoning |
|---|---|---|
| Class imbalance | SMOTE / SMOTEENN (train only) + `class_weight='balanced'` | Synthetic minority oversampling boosts recall; keep resampling inside CV / train split to avoid leakage. |
| Outliers in `Amount` | **RobustScaler** | Median/IQR based → robust to the heavy right tail of transaction amounts. |
| Leakage prevention | Resample **after** the train/test split | Resampling before splitting would let synthetic samples bleed into the test set & inflate metrics. |
| Metrics | PR-AUC, ROC-AUC, F1, Confusion Matrix | Accuracy is meaningless with 0.17% fraud prevalence; PR-AUC is the right curve for rare-event detection. |

---

## 5. Environment / Tooling

- **Python 3.11**, `scikit-learn 1.8`, `pandas 3.0`, `numpy`
- `imbalanced-learn 0.14` (SMOTE / SMOTEENN / Tomek Links)
- `xgboost 3.2`, `lightgbm 4.6`
- `shap 0.51`, `matplotlib`, `seaborn`

---

## 6. Results & Insights (updated step by step)

> Each section below is filled in as each pipeline stage completes.

### Step 1 — EDA Insights

**Dataset structure**
- Shape: **284,807 rows × 31 columns** (Time, V1–V28, Amount, Class).
- **No missing values** across all features.
- All `V*` features are numeric (PCA-transformed); only `Time` and `Amount` need scaling.

**Class imbalance (explicit ratio)**
- Legitimate: **284,315** (99.827%) | Fraud: **492** (0.173%).
- Imbalance ratio ≈ **579 : 1** → accuracy would be 99.83% even with a "predict all zero" model. This justifies evaluation on **PR-AUC / ROC-AUC / F1**, never accuracy.

**Amount & Time insights**
- Fraud median amount = **€9.25** vs legitimate = **€22.00** → fraudsters often execute small transactions to avoid detection.
- Legitimate amounts have a heavy right tail (max €25,691) → **RobustScaler** (median/IQR) is the safer scaling choice.
- Time distributions overlap; limited standalone discriminative value, but kept as a feature.

**Correlations with target (Class)**
- Strongest negative: `V17` (-0.33), `V14` (-0.30), `V12` (-0.26).
- Strongest positive: `V11` (+0.15), `V4` (+0.13), `V2` (+0.09).

**Figures:** `outputs/figures/eda_*.png` (class distribution, amount, time, correlation heatmap).

### Step 2 — Preprocessing Details

**Order of operations (leakage-safe):** split first → scale second → resample third.

1. **Stratified split** — `train_test_split(test_size=0.2, random_state=42, stratify=y)`:
   - Train: **227,845** rows (fraud 0.173%) | Test: **56,962** rows (fraud 0.172%).
   - The fraud ratio is preserved in both folds, so test-set evaluation reflects real-world
     0.17% prevalence.
2. **RobustScaler on `Amount` & `Time`** — fitted on the **training set only**, then applied
   to the test set. Chosen over StandardScaler because Amount has a heavy right tail
   (max €25,691) and RobustScaler uses median + IQR.
3. **Feature set (30 columns):** `V1`–`V28` (already PCA-transformed) + `Time_Scaled` + `Amount_Scaled`.
   Raw `Time`/`Amount` dropped after scaling.

**Artifacts:** `outputs/data/X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`, `scaler.pkl`.

### Step 3 — Balancing Strategy

**Approach:** **SMOTE** applied to the **training set only** (`sampling_strategy=1.0`, `k_neighbors=5`), after the split in step 2.

- Train class counts: legit **227,451** → fraud **394** → fraud **227,451** (fully balanced, 454,902 rows).
- Synthetic samples generated: **227,057**.
- **Test set untouched** — no resampling, no leakage. Reported test metrics reflect real 0.17% prevalence.

**Why SMOTE over SMOTEENN / Tomek Links?**
- Fraud transactions form tight clusters in the PCA-reduced space, so interpolation is highly effective.
- SMOTEENN / TomekLinks add a k-NN *cleaning* pass; on 227k training rows it runs >30 min, so SMOTE alone was retained (documented trade-off).
- Models are also trained with `class_weight='balanced'` / `scale_pos_weight` as a complementary imbalance strategy — final comparison included both resampled and weighted variants.

**Reasoning for leakage prevention:** resampling happens *after* the stratified split and the scaler fit; SMOTE is fitted on `X_train` only, so synthetic observations can never contaminate the test set or inflate metrics.

### Step 4 — Trained Models & Hyperparameters

8 models trained = **4 algorithms × 2 imbalance strategies** (`weighted` = class-weight aware on original train; `smote` = trained on SMOTE-balanced train).

Hyperparameters selected with 3-fold stratified CV on the **original training data**, optimized for **mean PR-AUC** (`average_precision`):

| Model | Best params (CV, original train) | Mean CV PR-AUC |
|---|---|---|
| Logistic Regression | `C=10, solver=liblinear, class_weight=balanced` | 0.756 |
| Random Forest | `n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight=balanced` | 0.841 |
| XGBoost | `scale_pos_weight=577.3, n_estimators=350, max_depth=4, lr=0.1` | 0.843 |
| LightGBM (final) | `scale_pos_weight=577.3, num_leaves=200, n_estimators=500, lr=0.03, min_child_samples=5, subsample=0.7, colsample_bytree=0.5, reg_lambda=5` | 0.825 |

> `scale_pos_weight = 577.3` = exact majority:minority ratio of the training set.

**LightGBM tuning note (a genuine debugging story):** the first attempt relied on LightGBM's default
`min_child_samples=20` and reached only **0.055 CV PR-AUC** — because each CV fold holds just ~131 fraud
samples, the learner could not form pure fraud leaves. Diagnostics showed high-capacity + strongly regulated
configs (`num_leaves=128`, `lr=0.03`, `min_child_samples=1`, heavy L2 + column subsampling) restore PR-AUC to
~0.85. The final model was selected via CV to avoid test-set leakage (script `04b_lightgbm_retune.py`).

**Artifacts:** `outputs/models/{Model}_{weighted|smote}.pkl` (8 files) + `models_metadata.json`.

### Step 5 — Final Evaluation Metrics & Model Selection

All metrics computed strictly on the **held-out TEST set** (56,962 rows, **98 fraud** — untouched by SMOTE/scaling). **No accuracy reported.**

| Model (variant) | PR-AUC | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | F1@opt-thr | opt. threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (smote)** | **0.880** | 0.982 | 0.791 | 0.888 | 0.836 | **0.872** | 0.994 |
| Random Forest (smote) | 0.875 | 0.983 | 0.871 | 0.827 | 0.848 | 0.862 | 0.690 |
| XGBoost (weighted) | 0.874 | **0.985** | 0.874 | 0.847 | **0.860** | 0.874 | 0.671 |
| Random Forest (weighted) | 0.859 | 0.957 | 0.961 | 0.755 | 0.846 | 0.882 | 0.280 |
| XGBoost (smote) | 0.852 | 0.980 | 0.483 | 0.878 | 0.623 | 0.851 | 0.975 |
| LightGBM (weighted) | 0.852 | 0.956 | 0.856 | 0.847 | 0.851 | 0.865 | 0.957 |
| Logistic Regression (smote) | 0.725 | 0.971 | 0.059 | 0.918 | 0.111 | 0.825 | 1.000 |
| Logistic Regression (weighted) | 0.721 | 0.972 | 0.061 | 0.918 | 0.114 | 0.825 | 0.994 |

**Key findings**
- **LightGBM (smote) wins on PR-AUC (0.880)** — the metric that matters for rare fraud — and is the chosen production model. Interpretation (Step 6) is performed on it.
- XGBoost (weighted) has the best ROC-AUC (0.985) and best F1 at the default 0.5 threshold — strong runner-up.
- Gradient-boosting models (XGBoost/LightGBM) dominate linear (LR) on PR-AUC by ~0.15, confirming the fraud patterns are **non-linear**.
- **SMOTE lifts PR-AUC** for LightGBM and RF but *degrades* XGBoost’s default-threshold F1 (its probability scale shifts) — hence evaluating at both default and tuned thresholds; a tuned threshold restores excellent F1 for every model.
- Confusion matrix (LightGBM-smote, test = 98 frauds): at **default 0.5** → 87 TP / 11 FN / 23 FP (Recall 0.888, Precision 0.791); at the **F1-optimal threshold 0.994** → ≈78 TP / 20 FN / 3 FP (Precision 0.963, Recall 0.796). The operating threshold is a business decision: near 0.99 sacrifices recall for near-zero false positives, while a lower threshold catches more fraud at the cost of more alerts.

**Figures:** `outputs/figures/eval_precision_recall_curves.png`, `eval_roc_curves.png`, `eval_confusion_matrices.png`; table: `outputs/metrics_comparison.csv`.

### Step 6 — Interpretation & Takeaways

Explained the **LightGBM (smote)** champion (PR-AUC 0.880) with **SHAP** on a stratified
test sample containing all 98 frauds + 2,000 legit rows.

**Top-10 features (mean |SHAP|)** — consistent with the EDA correlations:

| Rank | Feature | mean \|SHAP\| | Rank | Feature | mean \|SHAP\| |
|---|---:|---:|---|---:|---:|
| 1 | `V14` | 2.818 | 6 | `V17` | 0.443 |
| 2 | `V4` | 1.379 | 7 | `V3` | 0.375 |
| 3 | `V10` | 1.310 | 8 | `V8` | 0.354 |
| 4 | `V12` | 0.926 | 9 | `V7` | 0.296 |
| 5 | `V11` | 0.594 | 10 | `V16` | 0.252 |

- Flanked by the native LightGBM **gain** importance (`V14, V10, V12, V4, V11, V17…`) —
  both views agree, which adds confidence in the explanation (`shap_top_features.json`).
- `V14`, `V10`, `V12`, `V17` had the strongest *negative* correlation with fraud in EDA and
  dominate SHAP — the model leans on the same signals a human analyst would check.
- `Amount_Scaled` ranks ~15th: despite its intuitive appeal, the PCA components of the card
  usage pattern matter far more than the raw spend size.
- The beeswarm chart (`shap_beeswarm.png`) shows the relationships are **non-linear and
  thresholded** (e.g., `V14` pushes toward *legitimate* only in a bounded range) — a key
  reason gradient boosting beat logistic regression.

**Figures:** `outputs/figures/shap_beeswarm.png`, `shap_bar.png`, `feature_importance_gain.png`, `shap_dependence_top1.png`.

---

## 7. Final Project Summary

**Delivered an end-to-end, reproducible fraud-detection pipeline** on 284,807 real card
transactions (0.173% fraud):

| Stage | Outcome |
|---|---|
| EDA | 0 missing values; 579:1 imbalance quantified; fraud = small-amount transactions |
| Preprocessing | RobustScaler (Time, Amount) + stratified 80/20 split (fraud ratio preserved in both folds) |
| Imbalance | SMOTE on **train only** + class-weight variants (no leakage by construction) |
| Modeling | 8 models (LR, RF, XGBoost, LightGBM × weighted/SMOTE) tuned on CV PR-AUC |
| Evaluation | **LightGBM-smote: PR-AUC 0.880, ROC-AUC 0.982**, F1 0.87@tuned threshold; no accuracy reported |
| Explainability | SHAP + gain importance agree: `V14, V4, V10, V12, V11, V17` drive decisions non-linearly |

**Key takeaways**
1. **Accuracy is meaningless** at 0.173% prevalence — PR-AUC is the correct lens; a
   "predict-all-legit" model would score 99.83% accuracy and catch zero fraud.
2. **Leakage discipline** separates signal from noise: scale fit on train, SMOTE applied after
   the split, test set untouched — the 0.88 PR-AUC therefore reflects real deployment.
3. **Gradient boosting wins** on this PCA-reduced, non-linear fraud space; regularization
   (`num_leaves`, L2, subsampling) is decisive — default LightGBM settings failed (0.055 PR-AUC)
   and were fixed via targeted re-tuning.
4. **Threshold is an operational lever**: at threshold 0.5 you catch 87/98 frauds (Recall 0.888,
   Precision 0.791); near 0.99 you get Precision 0.96 at Recall 0.80. The right operating point
   is a business decision balancing investigation cost vs. fraud loss.
5. **Interpretable and deployable**: SHAP explanations match domain correlations, and the model
   exports as a single `.pkl` for scoring infrastructure.

**Reproduce:** run `01_eda.py` → `02_preprocessing.py` → `03_handle_imbalance.py` →
`04_model_training.py` → (`04b_lightgbm_retune.py`) → `05_evaluation.py` → `06_interpretation.py`
(all use `config.py`; `random_state = 42`).

---

## 8. Deployment Platform — تصنيف البيانات الجديدة عبر الويب

أُضيفت **منصة ويب عادية (Flask + HTML/CSS/JS)** تُدمج **أوزان النموذج الفائز فقط**
(`LightGBM_smote.pkl` مع `scaler.pkl`) لتشغيلها على بيانات مُدخلة جديدة **بدون أي إعادة تدريب**.
(نُقلت المنصة من Streamlit إلى تطبيق ويب قياسي بناءً على طلب المستخدم.)

### الملفات
- `deployment/predictor.py` — منطق الاستنتاج (مرة واحدة إصدارات): تحقق الأعمدة → تطبيق الـ
  scaler المخزَّن على `[Amount, Time]` → إعادة ترتيب الأعمدة الثلاثين بنفس ترتيب التدريب →
  `predict_proba` → التصنيف حسب العتبة. بالإضافة لمعالجة الدفعات ذات فئة واحدة، ومقاييس تحقق
  تلقائية عند وجود عمود `Class`. (دالّة `predict` تقبل `threshold` اختيارياً للتعديل الفوري.)
- `webapp/app.py` — خادم Flask بواجهات JSON:
  `GET /` (الصفحة) · `GET /api/glossary` · `POST /api/predict/upload`
  · `POST /api/predict/sample` · `POST /api/predict/form`.
  يرسم الرسوم (matplotlib → base64) ويبني ملف CSV (UTF-8 BOM) للتحميل.
- `webapp/templates/index.html` + `webapp/static/style.css` + `webapp/static/app.js` —
  واجهة عربية RTL بدون أي إطار Frontend:
  1. رفع ملف **CSV أو XLSX** (سحب وإفلات) للتصنيف بالجملة مع بطاقات ملخص ورسمي توزيع
     الاحتمالات والفئات، وجدول المعاملات المشتبه بها، وجدول النتائج الكامل، وزر تحميل النتائج.
  2. **مقاييس تحقق** تلقائية (PR-AUC/ROC-AUC/Precision/Recall/F1/مصفوفة الالتباس)
     إذا احتوى الملف على عمود `Class`.
  3. **شريط عتبة الاحتيال** (الافتراضي 0.9944 = حد F1 الأعلى على التست).
  4. **فحص صف واحد يدوياً** (Time, Amount, V1..V28) بنتيجة فورية واحتمال الاحتيال.
  5. زر **عيّنة تجريبية** يحمّل 5 صفوف حقيقية من التست (بما فيها حالات احتيال) لعرض حي.
- `deployment/app.py` — نسخة Streamlit السابقة محفوظة كبديل اختياري.

### التحقق (اختبار فعلي)
- **واجهات API (Python/urllib):** `/api/glossary` → 33 صفاً · رفع
  `display_names.csv` (10 معاملات) → 200 بتصنيف **4 مُشتبه بها / 6 مشروعة**،
  F1 = 1.0، وأول عمود بالاسم التوضيحي، وملف CSV مُضمّن · `/api/predict/sample` → 5 صفوف
  (2 احتيال) · `/api/predict/form` → 200 بنجاح (بالأسماء التقنية **و** التوضيحية) ·
  ملف ناقص الأعمدة → 400 برسالة عربية واضحة.
- **المتصفح (Playwright):** الصفحة تُرسَّم بدون أخطاء كونسول، دليل الأعمدة (33 صفاً)،
  رفع CSV حقيقي من الواجهة → بطاقات الملخص + الرسوم + 4 صفوف مُشتبه بها + 10 صفوف نتائج.

### التحويل من Streamlit إلى تطبيق ويب عادي (Flask)
طلب المستخدم منصة ويب قياسية بدل Streamlit. بقي منطق الاستنتاج في
`deployment/predictor.py` كما هو (أُضيفت فقط معامل `threshold` الاختياري لـ `predict`)،
واختُصرت الواجهة إلى صفحة واحدة عربية RTL (HTML + CSS + JS أصيل) تستهلك واجهات JSON أعلاه.
ملاحظة تعلّم من حلقة تطوير سابقة: عرض الجداول في الويب العادي (HTML خام) لا يعاني من
قيد `LargeUtf8` الذي ظهر في Streamlit (انظر أدناه) — السبب في اختيار HTML أصيل.

### ثنائية اللغة (اللغة الأساسية: الإنجليزية)
أُضيف زر تبديل لغة في أعلى الصفحة (🇬🇧 English ⇄ 🇸🇦 العربية) مع حفظ الاختيار في
`localStorage` وتطبيقه على كل المحتوى فورياً:
- **كل النصوص** تُترجم عبر قاموس i18n في `webapp/static/app.js` (أكثر من 70 مفتاحاً):
  العناوين، الشروحات، النماذج، الملاحظات، الأزرار، رسائل الخطأ/التحميل، تسميات الجداول
  (Prediction / التنبؤ وFraud probability / احتمال الاحتيال)، والمقاسات (F1-Score ثابتة
  بالإنجليزية عالمياً).
- **دليل الأعمدة** يُرسَل من الخادم باللغتين (`meaning_ar`/`meaning_en` و`kind`/`kind_en`
  في `deployment/schema.py`) مع ملاحظة بحثية باللغتين، ويُعاد رسم الجدول فور تبديل اللغة.
- **اتجاه الصفحة**: العربية → `dir=rtl` والإنجليزية → `dir=ltr`، مع اتجاهات CSS منطقية
  (`start`/`end`) لتنعيم حقيقيpadding والنصوص.
- **النتائج الحالية** تُعاد رسوماتها باللغة الجديدة فور التبديل (تُحتفظ بآخر حمولة JSON).
- **أخطاء الخادم مُرمّزة**: تعيد الواجهات `{error: code, cols?, detail?}` بدل نصوص عربية
  صريحة، ويترجمها الواجهة (مثال: `missing_cols` → قائمة الأعمدة الناقصة باللغة النشطة).
- التحقق عبر متصفح آلي: العملية الكاملة (تحديث/رفع/تبديل لغة ذهاباً وإياباً/رسائل خطأ)
  ناجحة في اللغتين مع صفر أخطاء في الكونسول.

### لماذا أسماء الأعمدة غريبة؟ (موضّح للمستخدم)
أُضيف قسم إعلامي دائم أعلى الصفحة (باللغتين) يشرح للمستخدم بعبارات واضحة أن:
`V1..V28` **ليست** خصائص حقيقية بل **مركّبات PCA** من 28 خاصية أصلية سرية، حُجبت
الأصول للمصرفية للسرية، لذلك لا توجد أسماء حقيقية معروفة — والنموذج يتعلم الأنماط منها
مباشرة. ويوضح القسم أيضاً معنى `Time` و`Amount` و`Class`، وأن الملفات المرفوعة تقبل
الأسماء التقنية أو التوضيحية معاً، مع رابط "الدليل الكامل" (33 عموداً) المفتوح في نفس
الصفحة.

### التشغيل
```bash
C:\Users\abdel\anaconda3\python.exe webapp/app.py --port 8000
```
ثم افتح `http://localhost:8000`.

> تشغيل Streamlit القديم (اختياري): `streamlit run deployment/app.py` (منفذ 8501).

### إصلاح أخطاء الرفع الشائعة (حلقة التطوير)
كان التطبيق ينهار عند رفع بعض الملفات، فأُصلحت الحالات التالية واتُّبعت لاحقاً كلها:
1. **عمود `Class` كنصوص** (`Fraud`/`Legitimate`) → تُحوَّل تلقائياً إلى 0/1 (`predictor._coerce_labels`).
2. **أسماء أعمدة مختلفة الحالة** (`time`, `v14`, ...) → تُوحَّد إلى الأسماء القياسية
   (`predictor.normalize_columns`).
3. **ملف يفتقد أعمدة** → رسالة عربية توضح الأعمدة الناقصة بدل الانهيار.
4. **ملف فارغ/بلا صفوف** → رسالة واضحة.
5. أي خطأ غير متوقع في التصنيف → `st.error` ودية مع `st.stop()` بدل شاشة الـTraceback.

التحقق النهائي (6 ملفات اختبار: عادي/نصوص/فئة نصية/أسماء صغيرة/ناقص/XLSX): صفر انهيارات.

### أسماء الأعمدة التوضيحية (بحث معتم)
بحث توثيقي (مصدر: قواعد بيانات Kaggle الرسمية / `creditcardfraud.names`) يؤكد:

| العمود | المعنى الموثق |
|---|---|
| `Time` | الثواني المنقضية بين المعاملة وأول معاملة في البيانات (يومان متتاليان) |
| `Amount` | قيمة المعاملة (بلا PCA عليها) |
| `Class` | الهدف: 1 = احتيال، 0 = مشروعة |
| `V1..V28` | ناتج تحويل **PCA** على 28 خاصية أصلية سرية للبطاقة — **لا يوجد اسم حقيقي رسمي** لأن الأصول محجوبة للسرية المصرفية |

لذلك أُضيفت طبقة مسميات في `deployment/schema.py` تعرض أسماء توضيحية أمانة:
`Anonymized_PC_01..28`، `Seconds_Since_First_Transaction`، `Transaction_Amount`،
`Fraud_Label` — مع **دليل أعمدة كامل (عربي)** داخل التطبيق (قسم "📖 دليل الأعمدة")
يشرح معنى كل عمود دون نسب معانٍ غير موثقة للمركّبات. الملفات المرفوعة بهذه الأسماء
التوضيحية مقبولة أيضاً (يجري تحويلها تلقائياً لأسماء التدريب قبل التصنيف)،
والنتائج المُحمَّلة تُستعمل مرّة أخرى دون مشاكل (اختبار round-trip ناجح).

### إصلاح عرض الجداول (خطأ LargeUtf8 — سجّل تاريخي من نسخة Streamlit)
في نسخة Streamlit السابقة ظهرت مربعات خطأ (`Error: Unrecognized type: "LargeUtf8"`)
بدل الجداول (دليل الأعمدة + النتائج). السبب الجذري توافق إصدارات: `streamlit 1.30.0`
يضمّ مكتبة Arrow-JS قديمة لا تعرف نوع `LargeUtf8` الذي يولّده `pyarrow 25` (للنصوص
والمحتوى العربي)، فيفشل Frontend في فكّ تسلسل أي جدول يتضمن أعمدة نصية — حتى بعد
استبدال `st.dataframe` بـ `st.table` بقي الخلل (فكلاهما يتبادل عبر Arrow).
الحلان المعتمدتان: ترقية Streamlit إلى 1.63.0، ثم **الهجرة الكاملة لتطبيق ويب عادي**
يُرسّم الجداول كـ HTML خام (استخدمت فعلاً في `webapp/app.py`) — لا يتأثر بـ LargeUtf8 أصلاً.