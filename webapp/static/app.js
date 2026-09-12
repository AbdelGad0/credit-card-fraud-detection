"use strict";

const $ = (id) => document.getElementById(id);

const thresholdInput = $("threshold");
const thresholdVal = $("thresholdVal");
const dropZone = $("dropZone");
const fileInput = $("fileInput");
const statusBar = $("statusBar");
const langToggle = $("langToggle");

let lang = localStorage.getItem("fraud_lang") || "en";
let lastPayload = null;
let lastSource = "";
let lastSourceIsSample = false;
let glossaryRows = [];
let glossaryNotes = { en: "", ar: "" };

// ---------------------------------------------------------------------------
// Translations
// ---------------------------------------------------------------------------
const I18N = {
  en: {
    "app.title": "🛡️ Credit Card Fraud Detection Platform",
    "app.subtitle": "Classify new transactions as Fraud / Legitimate through an " +
      "embedded LightGBM model (trained on 284,807 transactions — PR-AUC = 0.880) — no retraining needed.",
    "badge.model": "Model: LightGBM (SMOTE)",
    "lang.next": "العربية",

    "controls.title": "⚙️ Settings &amp; data source",
    "threshold.label": "Fraud threshold:",
    "threshold.help": "Transactions whose fraud probability exceeds this threshold are flagged as fraud.",
    "upload.drop": "Drag &amp; drop a CSV or XLSX file here, or",
    "upload.browse": "browse files",
    "upload.hint": "Technical headers (<code>Time, V1..V28, Amount</code>) or friendly ones " +
      "(<code>Seconds_Since_First_Transaction, Anonymized_PC_01…</code>) are both accepted.",
    "btns.sample": "🧪 Holdout test sample (5 rows)",
    "btns.single": "✍️ Check a single transaction manually",
    "sample.label": "holdout test set (5 real rows)",
    "single.title": "🧪 Manual single-transaction check",
    "single.time": "Time (seconds since first transaction)",
    "single.amount": "Amount (original currency)",
    "single.vHint": "V{i} — anonymized component #{n}",
    "single.analyze": "🔍 Analyze transaction",
    "single.fraud": "🚨 <b>Likely fraud</b> — fraud probability: <b>{p}</b> (threshold {t})",
    "single.legit": "✅ <b>Legitimate transaction</b> — fraud probability: <b>{p}</b> (threshold {t})",

    "cols.title": "❓ Why do the column names look strange?",
    "cols.pca": "The dataset was <b>anonymized by the issuing bank</b>: the columns " +
      "<code>V1–V28</code> are <b>not</b> real-world features. They are <b>Principal Components</b> " +
      "— a dimensionality-reduction transform over 28 original, confidential card-usage features. " +
      "The originals were withheld for confidentiality, so <b>no public real-world name exists</b> " +
      "for them. The model learns fraud patterns directly from these components.",
    "cols.time": "<code>Time</code> = seconds elapsed since the first transaction (two consecutive days are covered).",
    "cols.amount": "<code>Amount</code> = the transaction amount in the original currency.",
    "cols.class": "<code>Class</code> = the label (1 = fraud, 0 = legitimate) — used for training/evaluation only, never a model input.",
    "cols.upload": "Uploads accept the technical names above <b>or</b> friendlier display names " +
      "(e.g. <code>Anonymized_PC_01</code>, <code>Transaction_Amount</code>) — both work.",
    "cols.more": "📖 Full column glossary (33 columns)…",

    "results.title": "📊 Classification results",
    "source.label": "Data source: {src} — all required columns confirmed.",
    "source.truncated": "The table shows only the first {n} rows.",
    "source.csvTruncated": "The download includes only the first 50,000 rows.",
    "summary.total": "Total transactions",
    "summary.fraud": "Flagged as fraud 🔴",
    "summary.legit": "Legitimate 🟢",
    "summary.threshold": "Classification threshold",
    "metrics.pr": "PR-AUC",
    "metrics.roc": "ROC-AUC",
    "metrics.precision": "Precision",
    "metrics.recall": "Recall",
    "metrics.f1": "F1-Score",
    "metrics.confusion": "Confusion matrix: TP = <b>{tp}</b> · FP = <b>{fp}</b> · FN = <b>{fn}</b> · TN = <b>{tn}</b>",
    "metrics.singleClass": "This batch contains a single class, so discrimination metrics (PR-AUC / ROC-AUC) were disabled.",
    "metrics.failed": "Could not compute verification metrics on the Class column: {msg} — <b>(classification results are still shown)</b>",
    "metrics.none": "No <code>Class</code> column found — classification results are shown without verification metrics.",
    "flagged.title": "📋 Flagged transactions 🔴",
    "results.full": "📄 Full results",
    "dl.btn": "⬇️ Download results (CSV)",
    "flagged.empty": "No transactions flagged at this threshold.",
    "col.prediction": "Prediction",
    "col.prob": "Fraud probability",

    "dual.glossary.title": "📖 Column guide — what does each column mean?",
    "glossary.col": "Technical column",
    "glossary.display": "Friendly name",
    "glossary.kind": "Type",
    "glossary.meaning": "Meaning",

    "about.title": "ℹ️ Technical details about this platform",
    "about.model": "<b>Model:</b> <code>LightGBM_smote.pkl</code> (best: PR-AUC 0.880) + <code>scaler.pkl</code> (RobustScaler fitted on training data only).",
    "about.transform": "<b>Preprocessing:</b> <code>Amount</code> and <code>Time</code> are scaled with the same scaler and order as in training " +
      "(<code>[Amount, Time]</code>), then the 30 features are ordered as the model expects: <code>V1..V28, Time_Scaled, Amount_Scaled</code>.",
    "about.threshold": "<b>Default threshold:</b> 0.9944 (F1-optimal on the test set) — adjustable with the slider above.",
    "about.eval": "<b>Evaluation:</b> if the file contains a <code>Class</code> column, verification metrics are shown (PR-AUC / ROC-AUC / Precision / Recall / F1 / Confusion Matrix).",
    "about.src": "<b>Source:</b> <code>webapp/app.py</code> (server) + <code>deployment/predictor.py</code> (inference) + <code>deployment/schema.py</code> (friendly names &amp; glossary).",

    "foot": "Graduation project — Credit Card Fraud Detection · LightGBM",

    "err.no_file": "No file selected.",
    "err.unreadable": "Could not read the file: {detail}",
    "err.bad_extension": "The file must be in CSV or XLSX format.",
    "err.missing_cols": "Missing required columns: <b>{cols}</b>. Expected: Time, V1..V28, Amount (Class is optional).",
    "err.predict": "Could not classify the data: {detail}",
    "err.form_read": "Could not read the transaction data: {detail}",
    "err.unexpected": "Unexpected error: {detail}",
    "busy.upload": "Reading the file and classifying ({name})…",
    "busy.sample": "Loading the test sample and classifying…",
    "busy.form": "Analyzing the transaction…",
    "status.done": "Classification completed successfully."
  },

  ar: {
    "app.title": "🛡️ منصة كشف الاحتيال في المعاملات البنكية",
    "app.subtitle": "تصنيف المعاملات الجديدة كاحتيال / مشروعة عبر النموذج المدمج " +
      "LightGBM (مدرّب على 284,807 معاملة — PR-AUC = 0.880) — دون إعادة تدريب.",
    "badge.model": "النموذج: LightGBM (SMOTE)",
    "lang.next": "English",

    "controls.title": "⚙️ الإعدادات ومصدر البيانات",
    "threshold.label": "عتبة الاحتيال (Fraud threshold):",
    "threshold.help": "أي معاملة يتجاوز احتمال احتيالها هذه العتبة تُصنَّف احتيالاً.",
    "upload.drop": "اسحب ملف CSV أو XLSX هنا، أو",
    "upload.browse": "تصفّح الجهاز",
    "upload.hint": "الأسماء التقنية (<code>Time, V1..V28, Amount</code>) أو التوضيحية " +
      "(<code>Seconds_Since_First_Transaction, Anonymized_PC_01…</code>) كلاهما مقبول.",
    "btns.sample": "🧪 عيّنة تجريبية من التست (5 صفوف)",
    "btns.single": "✍️ فحص معاملة واحدة يدوياً",
    "sample.label": "مجموعة التست (5 صفوف حقيقية)",
    "single.title": "🧪 فحص معاملة واحدة يدوياً",
    "single.time": "Time (ثوانٍ من أول معاملة)",
    "single.amount": "Amount (المبلغ)",
    "single.vHint": "V{i} — المركّب المجهول #{n}",
    "single.analyze": "🔍 تحليل المعاملة",
    "single.fraud": "🚨 <b>احتيال مُحتمل</b> — احتمال الاحتيال: <b>{p}</b> (العتبة {t})",
    "single.legit": "✅ <b>معاملة مشروعة</b> — احتمال الاحتيال: <b>{p}</b> (العتبة {t})",

    "cols.title": "❓ لماذا تبدو أسماء الأعمدة غريبة؟",
    "cols.pca": "البيانات <b>مجهولة الهوية من البنك المُصدِر</b>: الأعمدة <code>V1–V28</code> " +
      "<b>ليست</b> خصائص حقيقية. إنها <b>مركّبات رئيسية (PCA)</b> — تحويل تقليل أبعاد على 28 خاصية " +
      "أصلية سرية لاستخدام البطاقة. الأصل محجوب لأسباب سرية، لذلك <b>لا يوجد اسم حقيقي رسمي</b> " +
      "لها — النموذج يتعلم أنماط الاحتيال مباشرة من هذه المركّبات.",
    "cols.time": "<code>Time</code> = الثواني المنقضية منذ أول معاملة (البيانات تغطي يومين متتاليين).",
    "cols.amount": "<code>Amount</code> = قيمة المعاملة بالعملة الأصلية.",
    "cols.class": "<code>Class</code> = الهدف (1 = احتيال، 0 = مشروعة) — يستخدم للتدريب/التقييم فقط، وليس مدخلاً للنموذج.",
    "cols.upload": "المرفوعات تقبل الأسماء التقنية أعلاه <b>أو</b> الأسماء التوضيحية " +
      "(مثل <code>Anonymized_PC_01</code>، <code>Transaction_Amount</code>) — كلاهما يعمل.",
    "cols.more": "📖 دليل الأعمدة الكامل (33 عموداً)…",

    "results.title": "📊 نتائج التصنيف",
    "source.label": "مصدر البيانات: {src} — تم تأكيد وجود جميع الأعمدة المطلوبة.",
    "source.truncated": "الجدول يعرض أول {n} صفوف فقط.",
    "source.csvTruncated": "ملف التنزيل يضم أول 50,000 صف فقط.",
    "summary.total": "إجمالي المعاملات",
    "summary.fraud": "مُشتبه باحتيالها 🔴",
    "summary.legit": "مشروعة 🟢",
    "summary.threshold": "عتبة التصنيف",
    "metrics.pr": "PR-AUC",
    "metrics.roc": "ROC-AUC",
    "metrics.precision": "Precision",
    "metrics.recall": "Recall",
    "metrics.f1": "F1-Score",
    "metrics.confusion": "مصفوفة الالتباس: TP = <b>{tp}</b> · FP = <b>{fp}</b> · FN = <b>{fn}</b> · TN = <b>{tn}</b>",
    "metrics.singleClass": "الدفعة تحتوي على فئة واحدة فقط — عُطِّلت مقاييس التمييز (PR-AUC / ROC-AUC).",
    "metrics.failed": "تعذر حساب مقاييس التحقق على عمود Class: {msg} — <b>(النتائج التصنيفية معروضة على أي حال)</b>",
    "metrics.none": "لا يوجد عمود <code>Class</code> — عُرِضت النتائج التصنيفية دون مقاييس تحقق.",
    "flagged.title": "📋 المعاملات المُشتبه بها 🔴",
    "results.full": "📄 النتائج الكاملة",
    "dl.btn": "⬇️ تحميل النتائج (CSV)",
    "flagged.empty": "لا توجد معاملات مُشتبه بها عند هذه العتبة.",
    "col.prediction": "التنبؤ",
    "col.prob": "احتمال الاحتيال",

    "dual.glossary.title": "📖 دليل الأعمدة — ماذا يعني كل عمود؟",
    "glossary.col": "العمود التقني",
    "glossary.display": "الاسم التوضيحي",
    "glossary.kind": "النوع",
    "glossary.meaning": "المعنى",

    "about.title": "ℹ️ تفاصيل تقنية حول المنصة",
    "about.model": "<b>النموذج:</b> <code>LightGBM_smote.pkl</code> (الأفضل: PR-AUC 0.880) + <code>scaler.pkl</code> (RobustScaler مبني على التدريب فقط).",
    "about.transform": "<b>المعالجة المسبقة:</b> يُقيَّس <code>Amount</code> و<code>Time</code> بنفس المُعايِر والترتيب كما في التدريب " +
      "(<code>[Amount, Time]</code>)، ثم تُرتَّب 30 خاصية كما يتوقع النموذج: <code>V1..V28, Time_Scaled, Amount_Scaled</code>.",
    "about.threshold": "<b>العتبة الافتراضية:</b> 0.9944 (توازن F1 على التست) — قابلة للتعديل من الشريط أعلاه.",
    "about.eval": "<b>التحقق:</b> إذا احتوى الملف على عمود <code>Class</code> تُعرض مقاييس التحقق (PR-AUC / ROC-AUC / Precision / Recall / F1 / مصفوفة الالتباس).",
    "about.src": "<b>المصدر:</b> <code>webapp/app.py</code> (الخادم) + <code>deployment/predictor.py</code> (المنطق) + <code>deployment/schema.py</code> (الأسماء التوضيحية والدليل).",

    "foot": "مشروع تخرج — كشف الاحتيال في بطاقات الائتمان · LightGBM",

    "err.no_file": "لم يتم اختيار ملف.",
    "err.unreadable": "تعذر قراءة الملف: {detail}",
    "err.bad_extension": "الملف يجب أن يكون بصيغة CSV أو XLSX.",
    "err.missing_cols": "أعمدة مطلوبة ناقصة: <b>{cols}</b>. المتوقع: Time, V1..V28, Amount (Class اختياري).",
    "err.predict": "تعذر تصنيف البيانات: {detail}",
    "err.form_read": "تعذر قراءة بيانات المعاملة: {detail}",
    "err.unexpected": "خطأ غير متوقع: {detail}",
    "busy.upload": "جاري قراءة الملف وتصنيف المعاملات ({name})…",
    "busy.sample": "جاري تحميل العيّنة من التست وتصنيفها…",
    "busy.form": "جاري تحليل المعاملة…",
    "status.done": "تم التصنيف بنجاح ✅"
  }
};

function t(key, params) {
  let s = (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key;
  if (params) {
    for (const k in params) s = s.replace(new RegExp("\\{" + k + "\\}", "g"), params[k]);
  }
  return s;
}

// ---------------------------------------------------------------------------
// i18n application
// ---------------------------------------------------------------------------
function applyTranslations() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.title = t("app.title").replace(/^[^a-zA-Z0-9\u0600-\u06FF]*/, "");
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.innerHTML = t(el.dataset.i18n);
  });
  langToggle.textContent = t("lang.next");
  buildVInputs();
  renderGlossary();
  if (lastPayload) renderResults(lastPayload, lastSource);
}

langToggle.addEventListener("click", () => {
  lang = lang === "en" ? "ar" : "en";
  localStorage.setItem("fraud_lang", lang);
  applyTranslations();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setStatus(msg, kind) {
  statusBar.classList.remove("hidden");
  statusBar.classList.remove("error", "ok", "busy");
  statusBar.classList.add(kind || "ok");
  statusBar.innerHTML = msg;
}

function clearStatus() {
  statusBar.classList.add("hidden");
}

function busy(msg) {
  statusBar.classList.remove("hidden", "error", "ok");
  statusBar.classList.add("busy");
  statusBar.textContent = msg;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function pct(p) {
  return (p * 100).toFixed(2) + "%";
}

function stat(lbl, val, sub, color) {
  return '<div class="stat ' + (color || "") + '"><div class="lbl">' + lbl +
    '</div><div class="val">' + esc(val) + "</div>" +
    (sub ? '<div class="sub">' + esc(sub) + "</div>" : "") + "</div>";
}

function headerLabel(col) {
  if (col === "Prediction") return t("col.prediction");
  if (col === "Fraud_Probability") return t("col.prob");
  return col;
}

function renderTable(el, rows, cols) {
  const head = "<thead><tr>" + cols.map((c) => "<th>" + headerLabel(c) + "</th>").join("") + "</tr></thead>";
  const body = rows.map((row) => {
    const cls = row.Prediction === "Fraud" ? ' class="Pred-fraud"' : "";
    return "<tr" + cls + ">" + cols.map((c) => "<td>" + esc(row[c]) + "</td>").join("") + "</tr>";
  }).join("");
  el.innerHTML = head + "<tbody>" + body + "</tbody>";
}

// ---------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------
async function loadGlossary() {
  try {
    const res = await fetch("/api/glossary");
    const data = await res.json();
    glossaryRows = data.rows;
    glossaryNotes = { ar: data.note_ar || "", en: data.note_en || "" };
    renderGlossary();
  } catch (e) {
    $("glossaryNote").textContent = "Glossary could not be loaded.";
  }
}

function renderGlossary() {
  const noteEl = $("glossaryNote");
  const tableEl = $("glossaryTable");
  const label = (key) => (lang === "ar" ? I18N.ar[key] : I18N.en[key]);
  noteEl.innerHTML = glossaryNotes[lang] || "";
  const rows = glossaryRows.map((r) => ({
    "column": r.column,
    "display": r.display,
    "kind": lang === "ar" ? r.kind : r.kind_en,
    "meaning": lang === "ar" ? r.meaning_ar : r.meaning_en,
  }));
  renderTable(tableEl, rows, ["column", "display", "kind", "meaning"]);
  // localize glossary table headers
  const heads = tableEl.querySelectorAll("thead th");
  const keys = ["glossary.col", "glossary.display", "glossary.kind", "glossary.meaning"];
  heads.forEach((th, i) => {
    if (keys[i]) th.textContent = label(keys[i]);
  });
}

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------
thresholdInput.addEventListener("input", () => {
  thresholdVal.textContent = parseFloat(thresholdInput.value).toFixed(4);
});

function currentThreshold() {
  return parseFloat(thresholdInput.value);
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
$("browseBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
dropZone.addEventListener("click", (e) => {
  if (e.target.id !== "browseBtn") fileInput.click();
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) submitUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    submitUpload(fileInput.files[0]);
    fileInput.value = "";
  }
});

async function submitUpload(file) {
  clearStatus();
  busy(t("busy.upload", { name: file.name }));
  const fd = new FormData();
  fd.append("file", file);
  fd.append("threshold", currentThreshold());
  const res = await fetch("/api/predict/upload", { method: "POST", body: fd });
  handleScoreResponse(res, file.name, false);
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------
$("sampleBtn").addEventListener("click", async () => {
  clearStatus();
  busy(t("busy.sample"));
  const fd = new FormData();
  fd.append("threshold", currentThreshold());
  const res = await fetch("/api/predict/sample", { method: "POST", body: fd });
  handleScoreResponse(res, "", true);
});

// ---------------------------------------------------------------------------
// Single-row form
// ---------------------------------------------------------------------------
const numFormat = (v) => ("00" + v).slice(-2);

function buildVInputs() {
  const wrap = $("vInputs");
  let html = "";
  for (let i = 1; i <= 28; i++) {
    html += "<label>" + t("single.vHint", { i: i, n: numFormat(i) }) +
      "<input type=\"number\" id=\"inp_V" + i + "\" step=\"0.000001\" value=\"0\" /></label>";
  }
  wrap.innerHTML = html;
}

$("singleBtn").addEventListener("click", () => $("singlePanel").classList.toggle("hidden"));

$("analyzeBtn").addEventListener("click", async () => {
  const ensure = (el) => (isNaN(parseFloat(el.value)) ? 0 : parseFloat(el.value));
  const row = { Time: ensure($("inp_Time")), Amount: ensure($("inp_Amount")) };
  for (let i = 1; i <= 28; i++) row["V" + i] = ensure($("inp_V" + i));
  $("singleResult").innerHTML = "";
  busy(t("busy.form"));
  const res = await fetch("/api/predict/form", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, threshold: currentThreshold() }),
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus(errMsg(data), "error");
    return;
  }
  clearStatus();
  const risk = data.prediction === "Fraud";
  const color = risk ? "var(--danger)" : "var(--accent)";
  const key = risk ? "single.fraud" : "single.legit";
  $("singleResult").innerHTML =
    '<div style="border:1px solid ' + color + ';color:' + color +
    ';padding:14px;border-radius:10px;">' +
    t(key, { p: pct(data.probability), t: data.threshold.toFixed(4) }) + "</div>";
});

// ---------------------------------------------------------------------------
// Shared response handling
// ---------------------------------------------------------------------------
async function handleScoreResponse(res, source, isSample) {
  const data = await res.json();
  if (!res.ok) {
    setStatus(errMsg(data), "error");
    return;
  }
  clearStatus();
  lastPayload = data;
  lastSource = source;
  lastSourceIsSample = isSample;
  renderResults(data, source);
  $("results").classList.remove("hidden");
  $("results").scrollIntoView({ behavior: "smooth" });
}

function errMsg(data) {
  const code = data.error || "unexpected";
  const detail = data.detail || "";
  switch (code) {
    case "no_file": return t("err.no_file");
    case "unreadable": return t("err.unreadable", { detail: esc(detail) });
    case "bad_extension": return t("err.bad_extension");
    case "missing_cols": return t("err.missing_cols", { cols: (data.cols || []).join(", ") });
    case "predict": return t("err.predict", { detail: esc(detail) });
    case "form_read": return t("err.form_read", { detail: esc(detail) });
    default: return t("err.unexpected", { detail: esc(code + (detail ? " — " + detail : "")) });
  }
}

function renderResults(data, source) {
  const meta = data.meta;
  const src = lastSourceIsSample ? t("sample.label") : (source || lastSource);
  const notes = [];
  notes.push(t("source.label", { src: src }));
  if (meta.tables_truncated) notes.push(t("source.truncated", { n: data.results.length }));
  if (meta.csv_truncated) notes.push(t("source.csvTruncated"));
  $("sourceLine").innerHTML = notes.join("&nbsp;&nbsp;·&nbsp;&nbsp;");

  const fraudPct = meta.total ? ((meta.fraud / meta.total) * 100).toFixed(2) : "0";
  $("summaryCards").innerHTML =
    stat(t("summary.total"), meta.total, "", "blue") +
    stat(t("summary.fraud"), meta.fraud, fraudPct + "%", "red") +
    stat(t("summary.legit"), meta.legit, "", "green") +
    stat(t("summary.threshold"), meta.threshold.toFixed(4), "");

  const g = data.metrics;
  if (g && g.error) {
    $("metricsCards").innerHTML = "";
    $("metricsNote").innerHTML = t("metrics.failed", { msg: esc(g.error) });
  } else if (g) {
    const v = (x) => (x === null || x === undefined ? "—" : x);
    $("metricsCards").innerHTML =
      stat(t("metrics.pr"), v(g.PR_AUC), "", "blue") +
      stat(t("metrics.roc"), v(g.ROC_AUC), "", "blue") +
      stat(t("metrics.precision"), v(g.Precision), "", "") +
      stat(t("metrics.recall"), v(g.Recall), "", "") +
      stat(t("metrics.f1"), v(g.F1), "", "green");
    $("metricsNote").innerHTML =
      (g.single_class_batch ? t("metrics.singleClass") + "<br>" : "") +
      t("metrics.confusion", { tp: g.TP, fp: g.FP, fn: g.FN, tn: g.TN });
  } else {
    $("metricsCards").innerHTML = "";
    $("metricsNote").innerHTML = t("metrics.none");
  }

  $("chartHist").src = "data:image/png;base64," + data.chart_hist;
  $("chartCat").src = "data:image/png;base64," + data.chart_cat;

  if (data.flagged.length) {
    const fcols = Object.keys(data.flagged[0]);
    renderTable($("flaggedTable"), data.flagged, fcols);
    $("flaggedTable").parentElement.classList.remove("hidden");
  } else {
    const emptyNote = "<tbody><tr><td style='color:var(--muted)' colspan='5'>" +
      t("flagged.empty") + "</td></tr></tbody>";
    $("flaggedTable").innerHTML = emptyNote;
    $("flaggedTable").parentElement.classList.remove("hidden");
  }

  const rcols = Object.keys(data.results[0] || {});
  renderTable($("resultsTable"), data.results, rcols);
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------
$("downloadBtn").addEventListener("click", () => {
  if (!lastPayload) return;
  const bin = atob(lastPayload.csv_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "fraud_classification_results.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---------------------------------------------------------------------------
applyTranslations();
loadGlossary();