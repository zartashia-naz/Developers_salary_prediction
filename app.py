"""
app.py — Salary Estimator Streamlit UI
Run: streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DevSalary — Engineer Salary Estimator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #0c0c0f; }
section[data-testid="stSidebar"] {
    background: #111116;
    border-right: 1px solid #1e1e28;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #111116;
    border: 1px solid #1e1e28;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
}
.metric-card.accent { border-color: #e8ff47; }
.metric-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555566;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #e8ff47;
    font-family: 'Syne', sans-serif;
    line-height: 1;
}
.metric-value.white { color: #f0f0f5; }
.metric-sub {
    font-size: 11px;
    color: #444455;
    margin-top: 4px;
}

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555566;
    border-bottom: 1px solid #1e1e28;
    padding-bottom: 8px;
    margin: 1.5rem 0 1rem;
}

/* Insight cards */
.insight-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1rem; }
.insight-card {
    background: #111116;
    border: 1px solid #1e1e28;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
.insight-card.positive { border-left: 3px solid #e8ff47; }
.insight-card.negative { border-left: 3px solid #ff4757; }
.insight-card.neutral  { border-left: 3px solid #3a86ff; }
.insight-title { font-size: 11px; color: #888899; margin-bottom: 4px; }
.insight-value { font-size: 14px; color: #f0f0f5; font-weight: 500; }

/* Driver chips */
.driver-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.driver-chip {
    padding: 4px 12px;
    border-radius: 2px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
}
.driver-chip.up   { background: #1a2a0a; color: #e8ff47; border: 1px solid #3a5a0a; }
.driver-chip.down { background: #2a0a0a; color: #ff6b6b; border: 1px solid #5a1a1a; }

/* Sidebar labels */
.sidebar-label {
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555566;
    margin: 1.2rem 0 0.4rem;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("C:\\Users\\User\\Desktop\\developer survey ML project\\training\\models\\model.pkl",         "rb") as f: model         = pickle.load(f)
    with open("C:\\Users\\User\\Desktop\\developer survey ML project\\training\\models\\scaler.pkl",        "rb") as f: scaler        = pickle.load(f)
    with open("C:\\Users\\User\\Desktop\\developer survey ML project\\training\\models\\feature_names.pkl", "rb") as f: feature_names = pickle.load(f)
    return model, scaler, feature_names

@st.cache_resource
def load_explainer(_model, _scaler, feature_names):
    df      = pd.read_csv("C:\\Users\\User\\Desktop\\developer survey ML project\\data\\survey_cleaned.csv")
    X       = df.drop(columns=["salary", "log_salary"])
    y       = df["log_salary"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_sc = pd.DataFrame(_scaler.transform(X_train), columns=feature_names)
    return shap.LinearExplainer(_model, X_train_sc), X_train_sc["log_work_exp"].mean()

@st.cache_data
def load_survey_stats():
    df = pd.read_csv("C:\\Users\\User\\Desktop\\developer survey ML project\\data\\survey_cleaned.csv")
    return {
        "mean_salary":   df["salary"].mean(),
        "median_salary": df["salary"].median(),
        "p25":           df["salary"].quantile(0.25),
        "p75":           df["salary"].quantile(0.75),
        "salaries":      df["salary"].values,
    }

model, scaler, FEATURES = load_artifacts()
explainer, _ = load_explainer(model, scaler, FEATURES)
stats        = load_survey_stats()

# ─────────────────────────────────────────────────────────────
# ENCODINGS
# ─────────────────────────────────────────────────────────────
ED_MAP = {
    "Secondary school": 2,
    "Some college (no degree)": 3,
    "Associate degree": 4,
    "Bachelor's degree": 5,
    "Master's degree": 6,
    "PhD / Professional": 7,
}
ORG_MAP = {
    "Freelancer / just me": 0,
    "Less than 20": 1,
    "20–99": 3,
    "100–499": 4,
    "500–999": 5,
    "1,000–4,999": 6,
    "5,000–9,999": 7,
    "10,000+": 8,
}
REMOTE_MAP = {"In-person": 0, "Hybrid": 1, "Remote": 2}

LABELS = {
    "years_code":      "Total yrs coding",
    "work_exp":        "Professional exp",
    "log_work_exp":    "Log(pro exp)",
    "education":       "Education",
    "org_size":        "Company size",
    "remote_work":     "Remote work",
    "is_manager":      "People manager",
    "is_fullstack":    "Full-stack dev",
    "is_backend":      "Back-end dev",
    "is_frontend":     "Front-end dev",
    "is_ml":           "ML / Data sci",
    "is_devops":       "DevOps / SRE",
    "is_mobile":       "Mobile dev",
    "lang_python":     "Python",
    "lang_rust":       "Rust",
    "lang_go":         "Go",
    "lang_typescript": "TypeScript",
    "lang_kotlin":     "Kotlin",
    "lang_scala":      "Scala",
    "lang_swift":      "Swift",
    "cloud_aws":       "AWS",
    "cloud_azure":     "Azure",
    "cloud_gcp":       "GCP",
    "web_react":       "React",
    "web_nextjs":      "Next.js",
    "web_fastapi":     "FastAPI",
    "web_django":      "Django",
    "uses_ai_tools":   "AI tools",
    "work_exp_sq":     "Exp² (curve)",
    "n_cloud":         "# cloud platforms",
    "n_hv_langs":      "# specialist langs",
    "n_webframes":     "# web frameworks",
    "exp_x_edu":       "Exp × Education",
    "exp_x_orgsize":   "Exp × Company size",
    "manager_x_exp":   "Manager × Exp",
    "ml_x_cloud":      "ML × Cloud breadth",
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 Your Profile")
    st.markdown("Fill in your details to get a salary estimate with full explanation.")
    st.divider()

    st.markdown('<div class="sidebar-label">Experience</div>', unsafe_allow_html=True)
    years_code = st.slider("Total years coding", 0, 40, 8, help="Including student/hobby years")
    work_exp   = st.slider("Professional experience (years)", 0, 40, 4)

    st.markdown('<div class="sidebar-label">Background</div>', unsafe_allow_html=True)
    education  = st.selectbox("Education", list(ED_MAP.keys()), index=3)
    org_size   = st.selectbox("Company size", list(ORG_MAP.keys()), index=4)
    remote     = st.selectbox("Work arrangement", list(REMOTE_MAP.keys()), index=1)

    st.markdown('<div class="sidebar-label">Role</div>', unsafe_allow_html=True)
    is_manager   = st.toggle("People Manager")
    col1, col2   = st.columns(2)
    with col1:
        is_fullstack = st.checkbox("Full-stack")
        is_backend   = st.checkbox("Back-end", value=True)
        is_frontend  = st.checkbox("Front-end")
    with col2:
        is_ml        = st.checkbox("ML / DS")
        is_devops    = st.checkbox("DevOps")
        is_mobile    = st.checkbox("Mobile")

    st.markdown('<div class="sidebar-label">Languages</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        lang_python = st.checkbox("Python", value=True)
        lang_rust   = st.checkbox("Rust")
        lang_go     = st.checkbox("Go")
        lang_ts     = st.checkbox("TypeScript")
    with col4:
        lang_kotlin = st.checkbox("Kotlin")
        lang_scala  = st.checkbox("Scala")
        lang_swift  = st.checkbox("Swift")

    st.markdown('<div class="sidebar-label">Cloud & Frameworks</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        cloud_aws   = st.checkbox("AWS")
        cloud_azure = st.checkbox("Azure")
        cloud_gcp   = st.checkbox("GCP")
        uses_ai     = st.checkbox("AI tools", value=True)
    with col6:
        web_react   = st.checkbox("React")
        web_nextjs  = st.checkbox("Next.js")
        web_fastapi = st.checkbox("FastAPI")
        web_django  = st.checkbox("Django")

# ─────────────────────────────────────────────────────────────
# BUILD FEATURE VECTOR
# ─────────────────────────────────────────────────────────────
ed_val     = ED_MAP[education]
org_val    = ORG_MAP[org_size]
remote_val = REMOTE_MAP[remote]

n_cloud    = int(cloud_aws) + int(cloud_azure) + int(cloud_gcp)
n_hv_langs = int(lang_rust) + int(lang_go) + int(lang_ts) + int(lang_kotlin) + int(lang_scala)
n_webframes= int(web_react) + int(web_nextjs) + int(web_fastapi) + int(web_django)

row = {
    "years_code":      float(years_code),
    "work_exp":        float(work_exp),
    "log_work_exp":    np.log1p(float(work_exp)),
    "education":       float(ed_val),
    "org_size":        float(org_val),
    "remote_work":     float(remote_val),
    "is_manager":      float(is_manager),
    "is_fullstack":    float(is_fullstack),
    "is_backend":      float(is_backend),
    "is_frontend":     float(is_frontend),
    "is_ml":           float(is_ml),
    "is_devops":       float(is_devops),
    "is_mobile":       float(is_mobile),
    "lang_python":     float(lang_python),
    "lang_rust":       float(lang_rust),
    "lang_go":         float(lang_go),
    "lang_typescript": float(lang_ts),
    "lang_kotlin":     float(lang_kotlin),
    "lang_scala":      float(lang_scala),
    "lang_swift":      float(lang_swift),
    "cloud_aws":       float(cloud_aws),
    "cloud_azure":     float(cloud_azure),
    "cloud_gcp":       float(cloud_gcp),
    "web_react":       float(web_react),
    "web_nextjs":      float(web_nextjs),
    "web_fastapi":     float(web_fastapi),
    "web_django":      float(web_django),
    "uses_ai_tools":   float(uses_ai),
    "work_exp_sq":     float(work_exp) ** 2,
    "n_cloud":         float(n_cloud),
    "n_hv_langs":      float(n_hv_langs),
    "n_webframes":     float(n_webframes),
    "exp_x_edu":       float(work_exp) * ed_val,
    "exp_x_orgsize":   float(work_exp) * org_val,
    "manager_x_exp":   float(is_manager) * float(work_exp),
    "ml_x_cloud":      float(is_ml) * n_cloud,
}

X_input    = pd.DataFrame([row])[FEATURES]
X_input_sc = pd.DataFrame(scaler.transform(X_input), columns=FEATURES)

pred_log    = float(model.predict(X_input_sc)[0])
pred_salary = np.exp(pred_log)
pred_salary = max(20_000, pred_salary)

shap_vals   = explainer(X_input_sc).values[0]

# Percentile calculation
percentile = int(np.mean(stats["salaries"] < pred_salary) * 100)

# Top driver (biggest positive SHAP feature)
sv_series   = pd.Series(shap_vals, index=FEATURES)
top_driver  = LABELS.get(sv_series.idxmax(), sv_series.idxmax())
top_drag    = LABELS.get(sv_series.idxmin(), sv_series.idxmin())

# ─────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2rem; color:#f0f0f5; margin-bottom:4px;'>
DevSalary <span style='color:#e8ff47'>_</span>
</h1>
<p style='color:#444455; font-size:12px; letter-spacing:0.08em; margin-bottom:2rem;'>
SOFTWARE ENGINEER SALARY ESTIMATOR · STACK OVERFLOW 2025 · US MARKET
</p>
""", unsafe_allow_html=True)

# ── TOP METRICS ROW ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card accent">
        <div class="metric-label">Estimated Salary</div>
        <div class="metric-value">${pred_salary:,.0f}</div>
        <div class="metric-sub">annual, USD</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Monthly (gross)</div>
        <div class="metric-value white">${pred_salary/12:,.0f}</div>
        <div class="metric-sub">before tax</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Percentile</div>
        <div class="metric-value white">{percentile}<span style='font-size:1rem'>th</span></div>
        <div class="metric-sub">vs survey respondents</div>
    </div>""", unsafe_allow_html=True)

with c4:
    vs_median = pred_salary - stats["median_salary"]
    sign      = "+" if vs_median >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">vs Median ($161k)</div>
        <div class="metric-value white" style='color:{"#e8ff47" if vs_median>=0 else "#ff6b6b"}'>{sign}${abs(vs_median):,.0f}</div>
        <div class="metric-sub">survey median salary</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN CONTENT — two columns ───────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ── LEFT: SHAP WATERFALL ─────────────────────────────────────
with left:
    st.markdown('<div class="section-head">Why this estimate?</div>', unsafe_allow_html=True)
    st.caption("Each bar shows how a feature pushes your salary above or below the dataset average of $150k")

    # Sort by absolute SHAP, take top 14
    top_n   = 14
    top_idx = np.argsort(np.abs(shap_vals))[-top_n:]
    sv_top  = shap_vals[top_idx]
    lb_top  = [LABELS.get(FEATURES[i], FEATURES[i]) for i in top_idx]

    # Convert SHAP (log scale) to approximate dollar impact
    baseline    = np.exp(explainer.expected_value)
    dollar_shap = np.exp(explainer.expected_value + sv_top) - baseline

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#111116")
    ax.set_facecolor("#111116")

    colors = ["#e8ff47" if v > 0 else "#ff4757" for v in sv_top]
    bars   = ax.barh(range(len(sv_top)), dollar_shap, color=colors,
                     height=0.6, alpha=0.9)

    ax.set_yticks(range(len(lb_top)))
    ax.set_yticklabels(lb_top, fontsize=10, color="#aaaaaa",
                       fontfamily="monospace")
    ax.axvline(0, color="#333344", linewidth=1)
    ax.set_xlabel("Approximate salary impact ($)", color="#555566",
                  fontsize=10, fontfamily="monospace")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:+.0f}k")
    )
    ax.tick_params(colors="#333344", labelsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#1e1e28", linewidth=0.6, linestyle="--")

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, dollar_shap)):
        if abs(val) > 500:
            sign = "+" if val >= 0 else ""
            ax.text(val + (200 if val >= 0 else -200), i,
                    f"{sign}${abs(val):,.0f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    color="#e8ff47" if val >= 0 else "#ff6b6b",
                    fontsize=8.5, fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(
        '<p style="font-size:11px;color:#333344;margin-top:4px;">'
        '🟡 Yellow = pushes salary higher &nbsp;|&nbsp; 🔴 Red = pushes salary lower'
        '</p>',
        unsafe_allow_html=True
    )

# ── RIGHT: INSIGHTS + DISTRIBUTION ───────────────────────────
with right:

    # Key insight cards
    st.markdown('<div class="section-head">Key Insights</div>', unsafe_allow_html=True)

    top3_pos = sv_series.nlargest(3)
    top3_neg = sv_series.nsmallest(2)

    pos_drivers = " · ".join([
        f"<span class='driver-chip up'>↑ {LABELS.get(f,f)}</span>"
        for f in top3_pos.index
    ])
    neg_drivers = " · ".join([
        f"<span class='driver-chip down'>↓ {LABELS.get(f,f)}</span>"
        for f in top3_neg.index
    ])

    st.markdown(f"""
    <div class="insight-card positive" style="margin-bottom:10px;">
        <div class="insight-title">BOOSTING YOUR SALARY</div>
        <div class="driver-row">{pos_drivers}</div>
    </div>
    <div class="insight-card negative" style="margin-bottom:10px;">
        <div class="insight-title">HOLDING SALARY BACK</div>
        <div class="driver-row">{neg_drivers}</div>
    </div>
    """, unsafe_allow_html=True)

    # Salary distribution
    st.markdown('<div class="section-head">Where You Land</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(5, 2.8))
    fig2.patch.set_facecolor("#111116")
    ax2.set_facecolor("#111116")

    ax2.hist(stats["salaries"], bins=50, color="#1e1e28",
             edgecolor="none", density=True)

    # Shade below prediction
    hist_vals, bin_edges = np.histogram(stats["salaries"], bins=50, density=True)
    for i in range(len(hist_vals)):
        color = "#e8ff47" if bin_edges[i] <= pred_salary else "#1e1e28"
        ax2.bar(bin_edges[i], hist_vals[i],
                width=bin_edges[i+1]-bin_edges[i],
                color=color, alpha=0.6, edgecolor="none", align="edge")

    ax2.axvline(pred_salary, color="#e8ff47", linewidth=2, zorder=5)
    ax2.axvline(stats["median_salary"], color="#333344",
                linewidth=1, linestyle="--")

    ax2.text(pred_salary, ax2.get_ylim()[1] * 0.85,
             f"  You\n  ${pred_salary/1000:.0f}k",
             color="#e8ff47", fontsize=8.5, fontfamily="monospace")

    ax2.set_xlabel("Annual Salary ($)", color="#555566",
                   fontsize=9, fontfamily="monospace")
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
    )
    ax2.set_yticks([])
    ax2.tick_params(colors="#333344", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=1)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.markdown(f"""
    <div style='display:flex; gap:8px; margin-top:8px;'>
        <div class='metric-card' style='padding:0.8rem 1rem; flex:1;'>
            <div class='metric-label'>25th pct</div>
            <div style='color:#f0f0f5; font-size:14px; font-weight:600;'>${stats['p25']/1000:.0f}k</div>
        </div>
        <div class='metric-card' style='padding:0.8rem 1rem; flex:1;'>
            <div class='metric-label'>Median</div>
            <div style='color:#f0f0f5; font-size:14px; font-weight:600;'>${stats['median_salary']/1000:.0f}k</div>
        </div>
        <div class='metric-card' style='padding:0.8rem 1rem; flex:1;'>
            <div class='metric-label'>75th pct</div>
            <div style='color:#f0f0f5; font-size:14px; font-weight:600;'>${stats['p75']/1000:.0f}k</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FULL FEATURE TABLE ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-head">Complete Feature Breakdown</div>', unsafe_allow_html=True)
st.caption("Every feature's contribution to your salary estimate, sorted by impact")

breakdown = pd.DataFrame({
    "Feature":      [LABELS.get(f, f) for f in FEATURES],
    "Your Value":   [f"{X_input[f].values[0]:.1f}" for f in FEATURES],
    "Impact ($)":   [f"{'+'if v>=0 else ''}${np.exp(explainer.expected_value + v) - np.exp(explainer.expected_value):,.0f}" for v in shap_vals],
    "SHAP (log)":   [f"{v:+.4f}" for v in shap_vals],
    "Direction":    ["↑ boost" if v > 0.001 else ("↓ drag" if v < -0.001 else "→ neutral") for v in shap_vals],
})
breakdown["_abs"] = np.abs(shap_vals)
breakdown = breakdown.sort_values("_abs", ascending=False).drop(columns=["_abs"])

st.dataframe(
    breakdown,
    hide_index=True,
    use_container_width=True,
    height=400,
    column_config={
        "Direction": st.column_config.TextColumn(width="small"),
        "Feature":   st.column_config.TextColumn(width="medium"),
    }
)

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='font-size:11px; color:#333344; text-align:center; font-family:DM Mono,monospace;'>
Ridge Regression · Stack Overflow Developer Survey 2025 · US respondents · $20k–$400k ·
Model R² = 0.33 · MAE ≈ $45k · Not financial advice.
</p>
""", unsafe_allow_html=True)