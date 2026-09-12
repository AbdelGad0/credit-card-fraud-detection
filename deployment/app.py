"""
app.py — Web platform (Streamlit) for predicting credit-card fraud on NEW data.

The app embeds the trained champion model (LightGBM-smote, PR-AUC 0.880) by
loading the saved weights + the train-fitted RobustScaler through
`predictor.FraudPredictor`. Users can:

  1. Upload a CSV / XLSX file of transactions → batch classification with
     risk summary, flagged-rows table, optional ground-truth metrics and
     downloadable results.
  2. Enter a single transaction manually → instant probability + verdict.
  3. Load a small REAL sample from the held-out test set for a live demo.

Run from the project root:
    streamlit run deployment/app.py
"""

import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictor import FraudPredictor, REQUIRED_COLS, GROUND_TRUTH_COL
from schema import display, GLOSSARY, SUMMARY_NOTE

# ---------------------------------------------------------------------------
# 0. Page config + RTL styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="منصة كشف الاحتيال", page_icon="🛡️", layout="wide",
)

st.markdown(
    """
    <style>
      .main, .block-container { direction: rtl; text-align: right; }
      .stDownloadButton, .stButton { direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_table(df: pd.DataFrame, max_rows: int = 1000):
    """
    Reliable table renderer.

    st.dataframe) is NOT used: on this Streamlit build its Arrow/GlideGrid fails
    with 'Unrecognized type: LargeUtf8' whenever a column holds long/non-ASCII
    strings, which surfaces an error box to the user. st.table renders a plain
    server-side HTML table and works everywhere. Very large frames are trimmed
    for display (download button always exports ALL rows).
    """
    if len(df) > max_rows:
        st.caption(f"عرض أول {max_rows:,} صفاً من أصل {len(df):,} (التحميل الكامل من الزر أدناه).")
        st.table(df.head(max_rows))
    else:
        st.table(df)

# ---------------------------------------------------------------------------
# 1. Load the model + scaler ONCE per session (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="جاري تحميل النموذج والمُعايِرات...")
def get_predictor() -> FraudPredictor:
    return FraudPredictor()


predictor = get_predictor()

threshold = predictor.default_threshold   # F1-optimal threshold (0.9944)


# ---------------------------------------------------------------------------
# 2. Sidebar — model info, threshold control, data source
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    st.markdown(
        "**النموذج المدموج:** LightGBM (SMOTE)\n\n"
        "PR-AUC: **0.880** · ROC-AUC: **0.982**"
    )

    threshold = st.slider(
        "عتبة الاحتيال (Fraud threshold)",
        min_value=0.0, max_value=1.0, value=float(predictor.default_threshold),
        step=0.01,
        help="الصفوف التي يتجاوز احتمال احتيالها هذه العتبة تُصنَّف احتيالاً.",
    )

    st.divider()
    st.subheader("📂 مصدر البيانات")
    use_sample = st.button("تحميل عيّنة تجريبية (5 صفوف حقيقية من التست)")
    uploaded = st.file_uploader(
        "ارفع ملف CSV أو XLSX للتصنيف بالجملة",
        type=["csv", "xlsx"],
    )

# ---------------------------------------------------------------------------
# 3. Main area — header
# ---------------------------------------------------------------------------
st.title("🛡️ منصة كشف الاحتيال في المعاملات البنكية")
st.caption(
    "تصنيف المعاملات الجديدة كاحتيال / مشروعة باستخدام LightGBM المدرب على "
    "284,807 معاملة (PR-AUC = 0.880) — بدون إعادة تدريب."
)


# ---------------------------------------------------------------------------
# 4. Read the active dataset (uploaded file / sample / None)
# ---------------------------------------------------------------------------
def read_upload(uploaded_file):
    """Parse the uploaded CSV/XLSX into a DataFrame."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, engine="openpyxl")


df_to_score = None
source_label = None

if use_sample:
    df_to_score = predictor.load_raw_sample(n=5, with_ground_truth=True)
    source_label = "عيّنة تجريبية من التست (مرفقة بعمود Class للتحقق)"
elif uploaded is not None:
    with st.spinner("قراءة الملف..."):
        try:
            df_to_score = predictor.normalize_columns(read_upload(uploaded))
        except Exception as exc:  # noqa: BLE001
            st.error(f"تعذر قراءة الملف: {exc}")
        else:
            source_label = f"`{uploaded.name}`"

# ---------------------------------------------------------------------------
# 5. Batch classification
# ---------------------------------------------------------------------------
if df_to_score is not None:
    missing = predictor._validate_columns(df_to_score)
    if missing:
        st.error(
            "⚠️ الملف ينقصه أعمدة مطلوبة: **" + ", ".join(missing) + "**\n\n"
            "الأعمدة المطلوبة: `Time, V1..V28, Amount` (عمود Class اختياري "
            "للتحقق من النتائج).\n\n"
            "ملاحظة: الأسماء التوضيحية مقبولة أيضاً — راجع **دليل الأعمدة**"
            " أسفل الصفحة.")
    else:
        st.info(f"مصدر البيانات: {source_label} — تم تأكيد وجود جميع الأعمدة المطلوبة.")
        st.caption(
            "تقبل المنصة أسماء الأعمدة التقنية (`Time, V1..V28, Amount`) أو "
            "الأسماء التوضيحية (`Seconds_Since_First_Transaction, "
            "Anonymized_PC_01, ...`) — دليل الأعمدة أسفل الصفحة.")

        with st.spinner("جاري تصنيف المعاملات..."):
            try:
                result = predictor.predict(df_to_score)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "⚠️ تعذر تصنيف الملف:\n\n"
                    f"`{exc}`\n\n"
                    "تأكد من أن الأعمدة `Time, V1..V28, Amount` موجودة وقيمها "
                    "أرقام (وليس نصوص)، وأعد المحاولة."
                )
                st.stop()

        # ---- Summary cards -------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("إجمالي المعاملات", len(result.table))
        c2.metric(
            "مُشتبه باحتيالها 🔴", result.fraud_count,
            delta=f"{100 * result.fraud_count / max(len(result.table), 1):.2f}%",
        )
        c3.metric("مشروعة 🟢", result.legit_count)
        c4.metric("عتبة التصنيف", f"{threshold:.3f}")

        # ---- Ground-truth evaluation (if Class column exists) ---------------
        if result.metrics:
            st.subheader("📊 تقييم على بيانات موسومة (Class متوفرة)")
            g = result.metrics
            if "error" in g:
                st.warning(
                    "تعذر حساب مقاييس التحقق على عمود Class: \n\n" + g["error"] +
                    "\n\n(عرضت النتائج التصنيفية على أي حال.)")
            elif g["single_class_batch"]:
                st.warning(
                    "الدفعة تحتوي على فئة واحدة فقط، لذا تُعرض مصفوفة الالتباس "
                    "والمقاييس المتاحة، وتُعطَّل مقاييس التمييز (PR-AUC / ROC-AUC).")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("PR-AUC", g["PR_AUC"] if g["PR_AUC"] is not None else "—")
            m2.metric("ROC-AUC", g["ROC_AUC"] if g["ROC_AUC"] is not None else "—")
            m3.metric("Precision",
                      g["Precision"] if g["Precision"] is not None else "—")
            m4.metric("Recall",
                      g["Recall"] if g["Recall"] is not None else "—")
            m5.metric("F1-Score", g["F1"])
            st.write(
                f"Confusion Matrix: **{g['TP']}** True Positive · "
                f"**{g['FP']}** False Positive · **{g['FN']}** False Negative · "
                f"**{g['TN']}** True Negative"
            )

        # ---- Charts ----------------------------------------------------------
        col_chart, col_table = st.columns([1, 2])

        with col_chart:
            st.subheader("📈 توزيع الاحتمالات")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(result.table["Fraud_Probability"], bins=40, color="#4C72B0")
            ax.axvline(threshold, color="red", ls="--", lw=1.5,
                       label=f"العتبة = {threshold:.3f}")
            ax.set_xlabel("احتمال الاحتيال"); ax.set_ylabel("العدد")
            ax.legend(); fig.tight_layout()
            st.pyplot(fig)

            st.subheader("🔎 الفئات")
            counts = result.table["Prediction"].value_counts().reindex(
                ["Fraud", "Legitimate"], fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
            ax2.bar(["احتيال", "مشروع"], counts.values,
                    color=["#C44E52", "#4C72B0"])
            ax2.set_ylabel("العدد"); fig2.tight_layout()
            st.pyplot(fig2)

        # Friendly (display) names for the human-facing tables/downloads
        disp_table = display(result.table)

        with col_table:
            st.subheader("📋 المعاملات المُشتبه بها 🔴")
            if len(result.flagged):
                show = display(result.flagged).drop(columns=["Fraud_Label"],
                                                   errors="ignore")
                col_order = ["Prediction", "Fraud_Probability"] + [
                    c for c in show.columns
                    if c not in ("Prediction", "Fraud_Probability")]
                show_table(show[col_order])
            else:
                st.success("لا توجد معاملات مشتبه بها عند هذه العتبة.")

            st.subheader("📄 النتائج الكاملة")
            show_table(disp_table)

        # ---- Download ---------------------------------------------------------
        csv = disp_table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ تحميل النتائج (CSV)", csv,
            file_name="fraud_classification_results.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# 6. Single-row manual check
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧪 فحص معاملة واحدة يدوياً")

with st.expander("أدخل قيم المعاملة (Time, Amount, V1..V28)"):
    cols = st.columns(3)
    raw_row = {}
    with cols[0]:
        raw_row["Time"] = st.number_input("Time (ثوانٍ من أول معاملة)",
                                          value=0.0, format="%.2f")
        raw_row["Amount"] = st.number_input("Amount (المبلغ)", value=0.0,
                                            format="%.2f")
    for i in range(1, 29):
        with cols[i % 3]:
            raw_row[f"V{i}"] = st.number_input(f"V{i}", value=0.0,
                                               format="%.6f")

    if st.button("🔍 تحليل المعاملة"):
        row_df = pd.DataFrame([raw_row])
        missing = predictor._validate_columns(row_df)
        if missing:
            st.error("أعمدة مفقودة: " + ", ".join(missing))
        else:
            res = predictor.predict(row_df)
            r = res.table.iloc[0]
            prob, verdict = r["Fraud_Probability"], r["Prediction"]
            if verdict == "Fraud":
                st.error(
                    f"🚨 **احتيال مُحتمل** — احتمال الاحتيال: "
                    f"{100 * prob:.2f}% (العتبة {threshold:.3f})")
            else:
                st.success(
                    f"✅ **معاملة مشروعة** — احتمال الاحتيال: "
                    f"{100 * prob:.2f}% (العتبة {threshold:.3f})")

# ---------------------------------------------------------------------------
# 7. Column glossary (research-backed names)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📖 دليل الأعمدة — ماذا يعني كل عمود؟", expanded=False):
    st.markdown("#### خلاصة البحث")
    st.warning(SUMMARY_NOTE)
    st.markdown(
        "نظرة بحثية على موثّق البيانات الرسمي تؤكّد أن `V1..V28` هي **مركّبات "
        "PCA** (تقليل أبعاد لـ28 خاصية أصلية سرية)، و`Time` = الثواني المنقضية "
        "منذ أول معاملة، و`Amount` = قيمة المعاملة، و`Class` = مؤشر الاحتيال. "
        "لا يوجد اسم حقيقي رسمي للمركّبات ذاتها بسبب السرية المصرفية."
    )
    st.markdown("#### التسميات الودية (Display Names)")
    mapping = pd.DataFrame(GLOSSARY).rename(
        columns={"column": "العمود التقني", "display": "الاسم التوضيحي",
                 "kind": "النوع", "meaning_ar": "المعنى"})
    show_table(mapping)

# ---------------------------------------------------------------------------
# 5b. About / how it works
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ تفاصيل تقنية حول المنصة"):
    st.markdown(
        """
- **النموذج:** `LightGBM_smote.pkl` (الأفضل: PR-AUC 0.880) و`scaler.pkl`
  (RobustScaler مبنِيّ على بيانات التدريب فقط — لا إعادة تدريب على الإطلاق).
- **التحويل:** تُحوَّل الحقول `Amount` و`Time` بنفس المُعايِر والترتيب المستخدم
  في التدريب (`[Amount, Time]`)، ثم ترتيب 30 عموداً: `V1..V28, Time_Scaled, Amount_Scaled`.
- **العتبة الافتراضية:** 0.9944 (توازن F1 على التست). تقدر تغيّرها من الشريط
  الجانبي حسب سياسة العمل.
- **الختبار:** عند وجود عمود `Class` في الملف تعرض المنصة مقاييس التحقق
  (PR-AUC / ROC-AUC / Precision / Recall / F1 / Confusion Matrix).
- **المصدر:** `deployment/predictor.py` (المنطق) + `deployment/app.py` (الواجهة).
        """
    )