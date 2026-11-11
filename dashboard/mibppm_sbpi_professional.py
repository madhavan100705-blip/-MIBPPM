from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MIBPPM | Blast Prediction",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sbpi"
MODEL_DIR = RESULTS / "saved_models"

FEATURES = [
    "UCS (MPa)",
    "RQD (%)",
    "Burden (m)",
    "Spacing (m)",
    "Powder Factor (kg/m³)",
]
TARGETS = [
    "P80 (mm)",
    "PPV (mm/s)",
    "Flyrock (m)",
    "Crusher Productivity (t/h)",
    "SBPI",
]
MODEL_FILES = {
    "P80 (mm)": "p80_mm_best_model.joblib",
    "PPV (mm/s)": "ppv_mm_s_best_model.joblib",
    "Flyrock (m)": "flyrock_m_best_model.joblib",
    "Crusher Productivity (t/h)": "crusher_productivity_t_h_best_model.joblib",
    "SBPI": "sbpi_best_model.joblib",
}
UNITS = {
    "P80 (mm)": "mm",
    "PPV (mm/s)": "mm/s",
    "Flyrock (m)": "m",
    "Crusher Productivity (t/h)": "t/h",
    "SBPI": "",
}

# ---- Visual system ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f4f6fa; }
    [data-testid="stHeader"] { background: rgba(244,246,250,.92); }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #17243a; letter-spacing: -0.035em; }
    h1 { font-size: 2.1rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
    p, label, span, div { }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e5eaf2; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 3px 12px rgba(20,35,60,.035);
    }
    [data-testid="stMetricLabel"] { color: #69778d !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #14233b !important; font-family: 'Manrope', sans-serif; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
        background: #fff; border-radius: 14px;
    }
    div.stButton > button {
        background: #183b63; color: white; border: 0; border-radius: 10px;
        padding: .65rem 1rem; font-weight: 700; min-height: 46px;
    }
    div.stButton > button:hover { background: #24517f; color: white; border: 0; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 9px !important; background: white;
    }
    [data-testid="stSidebar"] { background: #17283f; }
    [data-testid="stSidebar"] * { color: #eaf0f8 !important; }
    [data-testid="stSidebar"] [data-testid="stMetric"] { background: #223954; border-color: #38506c; }
    [data-testid="stTabs"] button { font-weight: 700; }
    .hero {
        background: linear-gradient(115deg, #142a45 0%, #1c456c 65%, #286080 100%);
        border-radius: 18px; padding: 25px 28px; color: white; margin-bottom: 22px;
        box-shadow: 0 8px 25px rgba(20,42,69,.13);
    }
    .hero-title { font-family: 'Manrope',sans-serif; font-size: 1.75rem; font-weight: 800; color: white; margin: 0; }
    .hero-sub { color: #c9d8e8; font-size: .94rem; margin-top: 5px; }
    .section-note { color: #728097; font-size: .88rem; margin-top: -7px; }
    .panel {
        background: white; border: 1px solid #e5eaf2; border-radius: 14px;
        padding: 17px 19px; margin-bottom: 12px;
    }
    .panel-label { color: #748198; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
    .panel-value { color: #17243a; font-family: 'Manrope',sans-serif; font-size: 1.55rem; font-weight: 800; margin-top: 4px; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    path = RESULTS / "cleaned_dataset.csv"
    df = pd.read_csv(path)
    return df.rename(columns={
        "Powder Factor (kg/mÂ³)": "Powder Factor (kg/m³)",
        "Powder Factor (kg/m3)": "Powder Factor (kg/m³)",
    })

@st.cache_resource
def load_models():
    loaded = {}
    for target, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        loaded[target] = joblib.load(path)
    return loaded

try:
    data = load_data()
    models = load_models()
except Exception as exc:
    st.error(f"Unable to load the dataset or saved models: {exc}")
    st.stop()

missing = [name for name in FEATURES if name not in data.columns]
if missing:
    st.error(f"Missing model input columns: {', '.join(missing)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⛏ MIBPPM")
    st.caption("BENCH BLASTING ANALYTICS")
    st.divider()
    st.markdown("**Dataset**")
    st.metric("Measured records", f"{len(data):,}")
    if "Blast ID" in data.columns:
        st.metric("Blast groups", f"{data['Blast ID'].nunique():,}")
    st.markdown("---")
    st.caption("SBPI model set · Grouped 5-fold validation")
    st.caption("Use predictions as research estimates, not operational blast-design values.")

# Header
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Bench Blasting Prediction</div>
      <div class="hero-sub">Mathematical and machine-learning analysis of blast performance parameters</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compact dataset range summary
with st.expander("Input parameter ranges", expanded=False):
    range_rows = []
    for feature in FEATURES:
        s = pd.to_numeric(data[feature], errors="coerce").dropna()
        range_rows.append({
            "Parameter": feature,
            "Minimum": round(float(s.min()), 4),
            "Maximum": round(float(s.max()), 4),
            "Unit": feature[feature.find("(")+1:feature.find(")")] if "(" in feature else "",
        })
    st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)

tab_run, tab_graphs, tab_validation = st.tabs(
    ["BLAST PARAMETERS", "SENSITIVITY GRAPHS", "MODEL PERFORMANCE"]
)

defaults = {
    "UCS (MPa)": 80.0,
    "RQD (%)": 55.0,
    "Burden (m)": 3.3,
    "Spacing (m)": 4.0,
    "Powder Factor (kg/m³)": 0.78,
}

with tab_run:
    left, right = st.columns([1.05, 1.45], gap="large")
    with left:
        st.markdown("### Input parameters")
        st.markdown('<div class="section-note">Enter the blast and rock properties below.</div>', unsafe_allow_html=True)
        values = {}
        for i, feature in enumerate(FEATURES):
            s = pd.to_numeric(data[feature], errors="coerce").dropna()
            lo, hi = float(s.min()), float(s.max())
            values[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=float(np.clip(defaults[feature], lo, hi)),
                step=max((hi - lo) / 100, 0.01),
                format="%.4f",
                key=f"main_{i}",
            )
        out_of_range = []
        for feature, value in values.items():
            s = pd.to_numeric(data[feature], errors="coerce").dropna()
            if value < s.min() or value > s.max():
                out_of_range.append(feature)
        if out_of_range:
            st.warning("Outside observed dataset range: " + ", ".join(out_of_range))
        run = st.button("CALCULATE PREDICTIONS", type="primary", use_container_width=True)

    with right:
        st.markdown("### Prediction results")
        if run:
            frame = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
            preds = {
                target: float(np.asarray(model.predict(frame)).reshape(-1)[0])
                for target, model in models.items()
            }
            st.session_state["mibppm_preds"] = preds
            st.session_state["mibppm_inputs"] = values

        if "mibppm_preds" in st.session_state:
            preds = st.session_state["mibppm_preds"]
            st.markdown("#### AI model")
            metric_rows = st.columns(2)
            for i, target in enumerate(TARGETS):
                col = metric_rows[i % 2]
                col.metric(target, f"{preds[target]:,.4f} {UNITS[target]}".strip())
            st.download_button(
                "Download result table",
                pd.DataFrame([{"Parameter": t, "AI model": preds[t], "Unit": UNITS[t]} for t in TARGETS]).to_csv(index=False).encode("utf-8"),
                "mibppm_predictions.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="panel"><div class="panel-label">Prediction output</div>'
                '<div class="panel-value">—</div><div class="section-note">Enter parameters and calculate to view results.</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Mathematical equation results")
    st.caption("Only verified equation outputs will appear here. The current SBPI dataset does not contain all required equation inputs.")
    equation_rows = [
        {"Parameter": "P80 (mm)", "Mathematical equation": "Requires bench height, charge mass, rock factor and uniformity inputs"},
        {"Parameter": "PPV (mm/s)", "Mathematical equation": "Requires charge per delay, monitoring distance, K and beta"},
        {"Parameter": "Flyrock (m)", "Mathematical equation": "Requires stemming, charge mass and site calibration"},
        {"Parameter": "Crusher Productivity (t/h)", "Mathematical equation": "Requires crusher and operating data"},
        {"Parameter": "SBPI", "Mathematical equation": "Requires the original index equation and weighting method"},
    ]
    st.dataframe(pd.DataFrame(equation_rows), use_container_width=True, hide_index=True)

with tab_graphs:
    st.markdown("### Parameter sensitivity")
    st.markdown('<div class="section-note">Change one input and view the response of each AI prediction.</div>', unsafe_allow_html=True)
    base = st.session_state.get("mibppm_inputs", defaults)
    selected = st.selectbox("Parameter", FEATURES)
    series = pd.to_numeric(data[selected], errors="coerce").dropna()
    observed_min, observed_max = float(series.min()), float(series.max())
    c1, c2, c3 = st.columns(3)
    with c1:
        low = st.number_input("From", min_value=0.0, value=observed_min, key=f"graph_low_{selected}")
    with c2:
        high = st.number_input("To", min_value=0.0001, value=observed_max, key=f"graph_high_{selected}")
    with c3:
        count = st.selectbox("Points", [20, 40, 60, 80, 100], index=1)
    if high <= low:
        st.error("The upper value must be greater than the lower value.")
    else:
        xs = np.linspace(low, high, count)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        for target, model in models.items():
            rows = []
            for x in xs:
                row = {f: float(base.get(f, defaults[f])) for f in FEATURES}
                row[selected] = float(x)
                rows.append(row)
            ax.plot(xs, model.predict(pd.DataFrame(rows, columns=FEATURES)), linewidth=2.2, label=target)
        ax.axvspan(observed_min, observed_max, alpha=.08, color="#7892ad", label="Observed range")
        ax.set_xlabel(selected)
        ax.set_ylabel("Predicted value (native units)")
        ax.set_title(f"Effect of {selected} on model predictions", pad=14, weight="bold")
        ax.grid(True, alpha=.22)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if low < observed_min or high > observed_max:
            st.warning("The selected graph range extends beyond the dataset used to train the models.")

with tab_validation:
    st.markdown("### Model performance")
    metrics_path = RESULTS / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        best = metrics.sort_values("MAE").groupby("Target", as_index=False).first()
        display = best[["Target", "Model", "MAE", "RMSE", "R2"]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Metrics are based on grouped out-of-fold predictions by Blast ID.")

    oof_path = RESULTS / "oof_predictions.csv"
    best_path = RESULTS / "best_models.json"
    if oof_path.exists() and best_path.exists():
        oof = pd.read_csv(oof_path)
        with open(best_path, encoding="utf-8") as f:
            best_info = json.load(f)
        target = st.selectbox("Output parameter", list(best_info.keys()))
        model_name = best_info[target]["best_model"]
        plot = oof[(oof["Target"] == target) & (oof["Model"] == model_name)]
        if not plot.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4.6))
                ax.scatter(plot["Actual"], plot["Predicted"], s=42, alpha=.8)
                mn = min(plot["Actual"].min(), plot["Predicted"].min())
                mx = max(plot["Actual"].max(), plot["Predicted"].max())
                ax.plot([mn, mx], [mn, mx], "--", linewidth=1.5)
                ax.set_xlabel("Measured")
                ax.set_ylabel("Predicted")
                ax.set_title("Measured vs predicted", weight="bold")
                ax.grid(alpha=.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(6, 4.6))
                ax.scatter(plot["Predicted"], plot["Residual"], s=42, alpha=.8)
                ax.axhline(0, linestyle="--", linewidth=1.5)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residual")
                ax.set_title("Prediction residuals", weight="bold")
                ax.grid(alpha=.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

st.markdown(
    '<div style="text-align:center;color:#8b96a8;font-size:.78rem;padding-top:25px;">MIBPPM · Academic research prototype</div>',
    unsafe_allow_html=True,
)
