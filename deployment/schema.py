"""
schema.py — Human-friendly column names + glossary for the fraud dataset.

Research summary (verified against the dataset documentation):
  * V1..V28 are the PRINCIPAL COMPONENTS obtained by PCA from 28 original,
    confidential card-use features. The original features are NOT public
    (confidentiality), so no authoritative 1:1 real-world meaning exists.
    The components can only be described honestly as 'anonymized PCA
    components'.
  * 'Time'   = seconds elapsed between the transaction and the FIRST
               transaction in the dataset (the set covers two consecutive days).
  * 'Amount' = transaction amount (original unit retained for cost-sensitive
               learning).
  * 'Class'  = target / response: 1 = fraud, 0 = legitimate.

This module maps every technical column to a clearer display name (used by the
web app for labelling tables and downloads) and provides the glossary that the
app shows to the user. Files uploaded with the display names are accepted too
(the predictor maps them back to the technical names that the model consumed).
"""

# ---------------------------------------------------------------------------
# 1. Technical -> display name mapping
# ---------------------------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    "Time": "Seconds_Since_First_Transaction",
    "Amount": "Transaction_Amount",
    "Time_Scaled": "Seconds_Since_First_Transaction_Scaled",
    "Amount_Scaled": "Transaction_Amount_Scaled",
    "Class": "Fraud_Label",
}
# The 28 PCA components: honestly labelled as anonymised principal components.
DISPLAY_NAMES.update(
    {f"V{i}": f"Anonymized_PC_{i:02d}" for i in range(1, 29)}
)

DEFAULT_COLUMNS = ("Time", "Amount", "Class") + tuple(f"V{i}" for i in range(1, 29))

# ---------------------------------------------------------------------------
# 2. Reverse map (display name / any common variant -> technical name).
#    Lets the app accept files that were labelled with the friendly names.
# ---------------------------------------------------------------------------
ALIAS_TO_TECHNICAL: dict[str, str] = {v: k for k, v in DISPLAY_NAMES.items()}


def display(df):
    """Return a copy of `df` whose columns carry the human-friendly names."""
    return df.rename(columns=DISPLAY_NAMES)


def technical(df):
    """Map (case-insensitive) display-variant headers back to technical names."""
    rename = {}
    aliases_lower = {k.lower(): v for k, v in ALIAS_TO_TECHNICAL.items()}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in aliases_lower and col != aliases_lower[key]:
            rename[col] = aliases_lower[key]
    return df.rename(columns=rename) if rename else df


# ---------------------------------------------------------------------------
# 3. Glossary for the UI (technical + display name + kind + AR/EN meanings)
# ---------------------------------------------------------------------------
GLOSSARY: list[dict] = []


def _add(raw, kind, kind_en, meaning_ar, meaning_en):
    GLOSSARY.append({
        "column": raw,
        "display": DISPLAY_NAMES.get(raw, raw),
        "kind": kind,
        "kind_en": kind_en,
        "meaning_ar": meaning_ar,
        "meaning_en": meaning_en,
    })


_add("Time", "وقت", "Time",
     "الوقت بالثواني المنقضي بين هذه المعاملة وأول معاملة في البيانات "
     "(البيانات تغطي يومين متتاليين).",
     "Time in seconds elapsed between this transaction and the first "
     "transaction in the dataset (the dataset covers two consecutive days).")
_add("Amount", "قيمة المعاملة", "Transaction amount",
     "قيمة المعاملة بالعملة الأصلية. لا توجد PCA عليها، لذا تُقيّس بشكل "
     "منفصل (RobustScaler).",
     "The transaction amount in the original currency. Not part of the PCA, "
     "so it is scaled separately (RobustScaler).")
_add("Class", "الهدف (المتغير المستجيب)", "Target (response variable)",
     "1 = معاملة احتيالية، 0 = معاملة مشروعة. تستخدم في التدريب والتقييم فقط، "
     "وليست مدخلاً للنموذج.",
     "1 = fraudulent transaction, 0 = legitimate. Used for training and "
     "evaluation only — never a model input.")

for i in range(1, 29):
    _add(f"V{i}", "مركّب PCA مجهول الهوية", "Anonymous PCA component",
         "المركّب رقم {i} من تحويل PCA على 28 خاصية أصلية للبطاقة. الخصائص "
         "الأصلية غير متاحة للعموم لأسباب سرية، لذا لا يوجد معنى حقيقي 1:1 "
         "معروف لهذه الأعمدة — يتعلم النموذج الأنماط منها كنسب متجهات.".format(i=i),
         f"Principal component #{i} of a PCA over 28 original confidential "
         "card features. The originals are not public for confidentiality, so "
         "no authoritative real-world meaning exists — the model learns "
         "patterns from these components directly.")

for raw in ("Time_Scaled", "Amount_Scaled"):
    _add(raw, "خاصية معالَجة (داخلية)", "Processed (internal) feature",
         "نسخة مُقيّسة تنتجها خطوة المعالجة المسبقة قبل الإدخال للنموذج "
         "(لا يرفعها المستخدم).",
         "Scaled version produced by the preprocessing step before entering "
         "the model (not uploaded by users).")

SUMMARY_NOTE = (
    "V1–V28 هي نتائج تحويل PCA (تقليل أبعاد) على **28 خاصية أصلية** لبيانات "
    "البطاقات الأوروبية. الأصل محجوب لأسباب سرية، لذلك لا يوجد اسم حقيقي رسمي "
    "لهذه الأعمدة — يُدرج هذا المشروع أسماء توضيحية (`Anonymized_PC_01` … ) "
    "مع التوضيح بأن معناها التداولي غير معلن، وهي الطريقة الأمنة علمياً "
    "بدلاً من نسب معانٍ غير موثقة."
)

SUMMARY_NOTE_EN = (
    "V1–V28 are the results of a PCA (dimensionality reduction) over 28 "
    "**original features** of European card transactions. The originals are "
    "withheld for confidentiality, so no authoritative real-world name exists "
    "for these columns. This project therefore uses honest display names "
    "(`Anonymized_PC_01` …) and clarifies that their everyday meaning is "
    "undisclosed — the scientifically safe approach, instead of attributing "
    "unverifiable meanings."
)