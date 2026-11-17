from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.engine.fragmentation import kuznesov_mean_fragment_size_cm

# -----------------------------------------------------------------------------
# MIBPPM | SBPI research dashboard — clean editorial / technical visual system
# Uses existing local SBPI dataset, saved pipelines, grouped OOF predictions.
# Does not modify training data, models, or experiment outputs.
# -----------------------------------------------------------------------------
st.set_page_config(page_title="MIBPPM | Research Dashboard", page_icon="⛏", layout="wide")
ROOT = PROJECT_ROOT
RESULTS = ROOT / "results" / "sbpi"
MODEL_DIR = RESULTS / "saved_models"
FRAGMENTATION_DATA = ROOT / "data" / "raw" / "caos_surface_confirmed.csv"

FEATURES = ["UCS (MPa)", "RQD (%)", "Burden (m)", "Spacing (m)", "Powder Factor (kg/m³)"]
TARGETS = ["P80 (mm)", "PPV (mm/s)", "Flyrock (m)", "Crusher Productivity (t/h)", "SBPI"]
MODEL_FILES = {
    "P80 (mm)": "p80_mm_best_model.joblib",
    "PPV (mm/s)": "ppv_mm_s_best_model.joblib",
    "Flyrock (m)": "flyrock_m_best_model.joblib",
    "Crusher Productivity (t/h)": "crusher_productivity_t_h_best_model.joblib",
    "SBPI": "sbpi_best_model.joblib",
}
UNITS = {"P80 (mm)": "mm", "PPV (mm/s)": "mm/s", "Flyrock (m)": "m", "Crusher Productivity (t/h)": "t/h", "SBPI": "index"}
FRAG_FEATURES = ["S_over_B", "H_over_B", "B_over_D", "T_over_B", "Pf_kg_m3", "XB_m", "E_GPa"]
FRAG_LABELS = {
    "S_over_B": "Spacing / burden",
    "H_over_B": "Bench height / burden",
    "B_over_D": "Burden / hole diameter",
    "T_over_B": "Stemming / burden",
    "Pf_kg_m3": "Powder factor (kg/m³)",
    "XB_m": "In-situ block size (m)",
    "E_GPa": "Young's modulus (GPa)",
}
FORMULA_LIMITATIONS = {
    "P80 (mm)": "Bench height, charge mass, rock factor, RWS, and a validated size-distribution method are required",
    "PPV (mm/s)": "Charge per delay, monitoring distance, and site-calibrated K and beta are required",
    "Flyrock (m)": "A selected, site-calibrated equation and its stemming and charge inputs are required",
    "Crusher Productivity (t/h)": "Crusher capacity and operating-condition inputs are required",
    "SBPI": "The original index equation, weights, and normalization are required",
}
DEFAULTS = {"UCS (MPa)": 80.0, "RQD (%)": 55.0, "Burden (m)": 3.3, "Spacing (m)": 4.0, "Powder Factor (kg/m³)": 0.78}

# Neutral graphite + restrained teal, with one warm accent for measured data.
INK = "#18252E"
MUTED = "#64747D"
GRID = "#E4E9E8"
TEAL = "#087E83"
TEAL_LIGHT = "#DCEFED"
AMBER = "#C58A35"
RED = "#B85C50"
PAPER = "#F5F7F6"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --teal:{TEAL}; --paper:{PAPER}; }}
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; color:var(--ink); }}
.stApp {{ background:var(--paper); }}
[data-testid="stHeader"] {{ background:rgba(245,247,246,.94); }}
.block-container {{ max-width:100%; padding:1.25rem 2.2rem .8rem 2.2rem; }}
h1,h2,h3 {{ font-family:'Manrope',sans-serif !important; color:{INK} !important; letter-spacing:-.035em; }}
h1 {{ font-size:2.05rem !important; font-weight:800 !important; }}
h2 {{ font-size:1.35rem !important; font-weight:750 !important; }}
h3 {{ font-size:1.04rem !important; font-weight:700 !important; }}
p, label {{ color:{INK}; }}
[data-testid="stSidebar"] {{ display:none; }}
section[data-testid="stMain"] > div {{ max-width:100%; }}
[data-testid="stMetric"] {{ background:#FFFFFF; border:1px solid #E2E8E6; border-radius:12px; padding:13px 16px; box-shadow:none; }}
[data-testid="stMetricLabel"] {{ color:{MUTED} !important; font-size:.78rem; font-weight:600; }}
[data-testid="stMetricValue"] {{ color:{INK} !important; font-family:'DM Mono',monospace; font-size:1.45rem; }}
[data-testid="stTabs"] button {{ font-weight:700; color:{MUTED}; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color:{TEAL} !important; border-bottom-color:{TEAL} !important; }}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{ border-radius:8px !important; background:#fff; }}
.stButton > button, .stDownloadButton > button {{ background:{TEAL}; color:#fff; border:0; border-radius:8px; font-weight:700; min-height:43px; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:#066A6E; color:#fff; border:0; }}
.hero {{ border-bottom:1px solid #DCE4E1; padding:12px 0 20px 0; margin-bottom:20px; }}
.eyebrow {{ color:{TEAL}; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:8px; }}
.hero-sub {{ color:{MUTED}; font-size:.95rem; margin-top:6px; max-width:780px; }}
.rule {{ border-top:1px solid #DCE4E1; margin:18px 0; }}
.kicker {{ color:{MUTED}; text-transform:uppercase; letter-spacing:.1em; font-size:.7rem; font-weight:700; }}
.small-note {{ color:{MUTED}; font-size:.83rem; }}
.output-card {{ background:#fff; border:1px solid #E1E8E5; border-radius:11px; padding:15px 17px; min-height:112px; }}
.output-name {{ color:{MUTED}; font-size:.78rem; font-weight:700; }}
.output-value {{ font-family:'DM Mono',monospace; color:{INK}; font-size:1.55rem; font-weight:500; margin-top:8px; }}
.output-unit {{ color:{MUTED}; font-size:.76rem; }}
/* High-contrast prediction page: explicit text colors for Streamlit themes */
[data-testid="stAppViewContainer"] .output-card,
[data-testid="stAppViewContainer"] .output-card * {{
    opacity:1 !important;
}}
[data-testid="stAppViewContainer"] .output-card .kicker,
[data-testid="stAppViewContainer"] .output-card .output-name,
[data-testid="stAppViewContainer"] .output-card .small-note {{
    color:{MUTED} !important;
}}
[data-testid="stAppViewContainer"] .output-card .output-value {{
    color:{INK} !important;
}}
[data-testid="stAppViewContainer"] .output-card .output-unit {{
    color:#52636B !important;
}}
[data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
[data-testid="stAppViewContainer"] [data-testid="stMetricValue"] {{
    color:{INK} !important;
}}
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown label {{
    color:{INK} !important;
}}
[data-testid="stAppViewContainer"] .stButton button,
[data-testid="stAppViewContainer"] .stDownloadButton button {{
    color:#FFFFFF !important;
}}

.status-chip {{ display:inline-block; color:{TEAL}; background:{TEAL_LIGHT}; border-radius:4px; padding:3px 7px; font-size:.68rem; font-weight:700; letter-spacing:.04em; }}
footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


def normalize_columns(df):
    return df.rename(columns={
        "Powder Factor (kg/mÂ³)": "Powder Factor (kg/m³)",
        "Powder Factor (kg/m3)": "Powder Factor (kg/m³)",
    })


@st.cache_data(show_spinner=False)
def load_data():
    p = RESULTS / "cleaned_dataset.csv"
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return normalize_columns(pd.read_csv(p))


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for target, fname in MODEL_FILES.items():
        p = MODEL_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Saved model not found: {p}")
        out[target] = joblib.load(p)
    return out


@st.cache_resource(show_spinner=False)
def load_fragmentation_model():
    if not FRAGMENTATION_DATA.exists():
        raise FileNotFoundError(f"Fragmentation training data not found: {FRAGMENTATION_DATA}")
    frame = pd.read_csv(FRAGMENTATION_DATA).dropna(subset=FRAG_FEATURES + ["x50_m"])
    if len(frame) < 2:
        raise ValueError("At least two measured X50 records are required to fit the comparison model.")
    estimator = LinearRegression().fit(frame[FRAG_FEATURES], frame["x50_m"])
    ranges = {feature: (float(frame[feature].min()), float(frame[feature].max())) for feature in FRAG_FEATURES}
    return estimator, ranges, len(frame)


def kuznesov_x50_m(rock_factor, burden_m, spacing_m, bench_height_m, powder_factor, rws):
    """Kuznetsov-style X50, converted from the engine's centimetre result to metres."""
    if min(rock_factor, burden_m, spacing_m, bench_height_m, powder_factor, rws) <= 0:
        raise ValueError("Rock factor, geometry, powder factor, and RWS must all be positive.")
    rock_volume = burden_m * spacing_m * bench_height_m
    charge_mass = powder_factor * rock_volume
    return kuznesov_mean_fragment_size_cm(rock_factor, rock_volume, charge_mass, rws) / 100.0


def numeric(s):
    return pd.to_numeric(s, errors="coerce").dropna()


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=13)
    ax.set_xlabel(xlabel, color=MUTED, labelpad=8)
    ax.set_ylabel(ylabel, color=MUTED, labelpad=8)
    ax.grid(axis="both", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(6))


def render_fig(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


try:
    data = load_data()
    models = load_models()
    frag_model, frag_ranges, frag_training_count = load_fragmentation_model()
except Exception as exc:
    st.error(f"Dashboard data/model loading failed: {exc}")
    st.info("Confirm that this file is inside the project's dashboard folder and that results/sbpi contains the dataset and saved_models folder.")
    st.stop()

missing = [f for f in FEATURES if f not in data.columns]
if missing:
    st.error("Required model input columns are missing: " + ", ".join(missing))
    st.stop()

# Main dashboard header
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">MIBPPM / BENCH BLAST PERFORMANCE</div>
  <h1>Research analysis console</h1>
  <div class="hero-sub">Compare the Kuznetsov equation with an AI fragmentation estimate, inspect parameter responses, and export calculation results.</div>
</div>
""", unsafe_allow_html=True)

# Shared summary
rcols = st.columns(4)
for col, label, val in zip(rcols, ["DATA RECORDS", "BLAST GROUPS", "MODEL INPUTS", "PREDICTED OUTPUTS"], [len(data), data["Blast ID"].nunique() if "Blast ID" in data else "—", len(FEATURES), len(TARGETS)]):
    col.markdown(f'<div class="output-card"><div class="kicker">{label}</div><div class="output-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
tab_predict, tab_formula, tab_sensitivity, tab_validation, tab_data = st.tabs(["PREDICTION", "FORMULA vs AI · X50", "PARAMETER RESPONSE", "MODEL VALIDATION", "DATA PROFILE"])

# ---- Prediction ----
with tab_predict:
    left, right = st.columns([.9, 1.4], gap="large")
    with left:
        st.markdown("### Blast and rock inputs")
        st.markdown('<p class="small-note">Enter values in the units shown. Values are not capped at the observed dataset limits.</p>', unsafe_allow_html=True)
        inputs = {}
        for i, feature in enumerate(FEATURES):
            vals = numeric(data[feature])
            lo, hi = float(vals.min()), float(vals.max())
            step = max((hi - lo) / 100.0, 0.01)
            inputs[feature] = st.number_input(
                feature, min_value=0.0, value=float(np.clip(DEFAULTS[feature], lo, hi)),
                step=step, format="%.4f", key=f"sbpi_v3_input_{i}"
            )
        extrapolated = [f for f, v in inputs.items() if v < numeric(data[f]).min() or v > numeric(data[f]).max()]
        if extrapolated:
            st.warning("Outside observed training range: " + ", ".join(extrapolated) + ". Predictions are extrapolations.")
        calculate = st.button("RUN MODEL PREDICTION", type="primary", use_container_width=True)
    with right:
        st.markdown("### Predicted blast performance")
        if calculate:
            x = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
            estimates = {t: float(np.asarray(models[t].predict(x)).reshape(-1)[0]) for t in TARGETS}
            st.session_state["sbpi_v2_estimates"] = estimates
            st.session_state["sbpi_v2_inputs"] = inputs
            st.session_state["sbpi_v2_run_id"] = pd.Timestamp.now().isoformat(timespec="seconds")
        estimates = st.session_state.get("sbpi_v2_estimates")
        if estimates:
            cards = st.columns(2)
            for i, target in enumerate(TARGETS):
                with cards[i % 2]:
                    st.markdown(f'<div class="output-card"><div class="output-name">{target}</div><div class="output-value">{estimates[target]:,.4f}</div><div class="output-unit">{UNITS[target]} · AI model estimate</div></div>', unsafe_allow_html=True)
                    st.write("")
            run_inputs = st.session_state.get("sbpi_v2_inputs", inputs)
            report_rows = []
            for target in TARGETS:
                report_rows.append({
                    "Run timestamp": st.session_state.get("sbpi_v2_run_id", ""),
                    **run_inputs,
                    "Output": target,
                    "AI model prediction": estimates[target],
                    "Unit": UNITS[target],
                    "Formula result": "",
                    "Formula status": FORMULA_LIMITATIONS[target],
                    "Model": "Saved target-specific Linear Regression pipeline",
                })
            csv = pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button("DOWNLOAD FULL CALCULATION · CSV", csv, "mibppm_calculation_report.csv", "text/csv")
            st.caption("The report includes every input, all five model estimates, units, and an explicit formula availability field.")
        else:
            st.markdown('<div class="output-card"><div class="kicker">MODEL OUTPUTS</div><div class="output-value">—</div><p class="small-note">Run prediction to populate the five output estimates.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("### Equation-based outputs")
    st.caption("The existing SBPI input table does not contain the full input set and calibration constants needed to calculate these independently. Unavailable equation values are intentionally left blank rather than estimated or invented.")
    math_status = pd.DataFrame([
        [target, "Not configured", FORMULA_LIMITATIONS[target]] for target in TARGETS
    ], columns=["Output", "Equation result", "Missing equation inputs"])
    st.dataframe(math_status, use_container_width=True, hide_index=True)

# ---- Shared-output mathematical and AI comparison ----
with tab_formula:
    st.markdown("### Kuznetsov equation vs AI estimate")
    st.markdown(
        '<p class="small-note">Compare two estimates of the same output: median fragment size X50 (m). '
        'The AI model uses the repository’s measured X50 dataset; the equation uses your stated rock factor and blast inputs.</p>',
        unsafe_allow_html=True,
    )
    with st.expander("Calculation inputs", expanded=True):
        a, b, c = st.columns(3)
        with a:
            frag_burden = st.number_input("Burden (m)", min_value=0.1, value=3.0, step=0.1, key="frag_burden")
            frag_spacing = st.number_input("Spacing (m)", min_value=0.1, value=3.6, step=0.1, key="frag_spacing")
            frag_bench = st.number_input("Bench height (m)", min_value=0.5, value=12.0, step=0.5, key="frag_bench")
            frag_stemming = st.number_input("Stemming length (m)", min_value=0.0, value=2.4, step=0.1, key="frag_stemming")
        with b:
            frag_hole_diameter = st.number_input("Hole diameter (mm)", min_value=25.0, value=100.0, step=5.0, key="frag_hole_diameter")
            frag_powder_factor = st.number_input("Powder factor (kg/m³)", min_value=0.01, value=0.8, step=0.05, key="frag_powder_factor")
            frag_block_size = st.number_input("In-situ block size XB (m)", min_value=0.001, value=0.5, step=0.01, key="frag_block_size")
            frag_modulus = st.number_input("Young’s modulus (GPa)", min_value=0.1, value=30.0, step=1.0, key="frag_modulus")
        with c:
            frag_rock_factor = st.number_input(
                "Documented rock factor A", min_value=0.0, max_value=30.0, value=0.0, step=0.1,
                help="Enter a factor justified for your rock mass. Zero leaves the equation comparison disabled.",
                key="frag_rock_factor",
            )
            frag_rws = st.number_input("Relative weight strength (RWS)", min_value=1.0, value=100.0, step=1.0, key="frag_rws")
            actual_x50 = st.number_input("Measured X50 (m), optional", min_value=0.0, value=0.0, step=0.01, key="frag_actual_x50")
            st.caption(f"AI training records: {frag_training_count} measured X50 rows")
        st.caption("RWS starts at 100 as an ANFO example; verify the explosive strength before calculating. The equation assumes one burden × spacing × bench-height volume per hole and uses charge mass = powder factor × volume.")

    if frag_rock_factor <= 0:
        st.info("Enter a positive, independently justified rock factor to calculate the equation estimate.")
    current_signature = (
        frag_rock_factor, frag_rws, frag_burden, frag_spacing, frag_bench,
        frag_stemming, frag_hole_diameter, frag_powder_factor, frag_block_size,
        frag_modulus, actual_x50,
    )
    run_fragmentation = st.button("CALCULATE X50 · FORMULA + AI", type="primary", key="run_x50_comparison")
    if run_fragmentation:
        try:
            if frag_stemming >= frag_bench:
                raise ValueError("Stemming must be shorter than bench height.")
            if frag_hole_diameter <= 0:
                raise ValueError("Hole diameter must be positive.")
            if frag_rock_factor <= 0:
                raise ValueError("Enter a positive, independently justified rock factor.")
            x50_math = kuznesov_x50_m(
                frag_rock_factor, frag_burden, frag_spacing, frag_bench,
                frag_powder_factor, frag_rws,
            )
            frag_features = {
                "S_over_B": frag_spacing / frag_burden,
                "H_over_B": frag_bench / frag_burden,
                "B_over_D": frag_burden / (frag_hole_diameter / 1000.0),
                "T_over_B": frag_stemming / frag_burden,
                "Pf_kg_m3": frag_powder_factor,
                "XB_m": frag_block_size,
                "E_GPa": frag_modulus,
            }
            frag_input = pd.DataFrame([frag_features], columns=FRAG_FEATURES)
            x50_ai = float(np.asarray(frag_model.predict(frag_input)).reshape(-1)[0])
            out_of_range_frag = [
                f"{FRAG_LABELS[k]} ({v:.3g})" for k, v in frag_features.items()
                if v < frag_ranges[k][0] or v > frag_ranges[k][1]
            ]
            st.session_state["x50_comparison"] = {
                "math": x50_math,
                "ai": x50_ai,
                "features": frag_features,
                "inputs": {
                    "Rock factor A": frag_rock_factor,
                    "Burden (m)": frag_burden,
                    "Spacing (m)": frag_spacing,
                    "Bench height (m)": frag_bench,
                    "Stemming (m)": frag_stemming,
                    "Hole diameter (mm)": frag_hole_diameter,
                    "Powder factor (kg/m³)": frag_powder_factor,
                    "RWS": frag_rws,
                    "In-situ block size (m)": frag_block_size,
                    "Young’s modulus (GPa)": frag_modulus,
                    "Measured X50 (m)": actual_x50 if actual_x50 > 0 else np.nan,
                },
                "out_of_range": out_of_range_frag,
                "signature": current_signature,
            }
        except Exception as exc:
            st.error(f"X50 calculation could not be completed: {exc}")

    comparison = st.session_state.get("x50_comparison")
    if comparison and comparison["signature"] != current_signature:
        comparison = None
    if comparison:
        if comparison["out_of_range"]:
            st.warning("Outside observed AI training range: " + ", ".join(comparison["out_of_range"]) + ". Treat the AI estimate as an extrapolation.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Kuznetsov-style equation · X50", f"{comparison['math']:.4f} m")
        m2.metric("AI estimate · X50", f"{comparison['ai']:.4f} m")
        m3.metric("AI − equation", f"{comparison['ai'] - comparison['math']:+.4f} m")
        st.caption("Equation form: X50 = A × (V/Q)^0.8 × Q^(1/6) × (115/RWS)^(19/30). V = burden × spacing × bench height; Q = powder factor × V. The code returns cm, converted here to m.")
        st.caption("The AI is a Linear Regression fitted to all measured X50 records in the included research dataset. It is a research estimate, not an independently validated site prediction.")

        sensitivity_rows = []
        base_features = comparison["features"]
        base_inputs = comparison["inputs"]
        for feature in FRAG_FEATURES:
            low, high = frag_ranges[feature]
            xgrid = np.linspace(low, high, 60)
            rows = []
            for xv in xgrid:
                row = dict(base_features)
                row[feature] = float(xv)
                rows.append(row)
            ai_curve = np.asarray(frag_model.predict(pd.DataFrame(rows, columns=FRAG_FEATURES)), dtype=float).reshape(-1)
            for xv, ai_value in zip(xgrid, ai_curve):
                burden = base_inputs["Burden (m)"]
                spacing = base_inputs["Spacing (m)"]
                bench = base_inputs["Bench height (m)"]
                pf = base_inputs["Powder factor (kg/m³)"]
                if feature == "S_over_B":
                    spacing = float(xv) * burden
                elif feature == "H_over_B":
                    bench = float(xv) * burden
                elif feature == "Pf_kg_m3":
                    pf = float(xv)
                formula_value = kuznesov_x50_m(
                    base_inputs["Rock factor A"], burden, spacing, bench, pf, base_inputs["RWS"]
                )
                sensitivity_rows.append({
                    "Varied parameter": FRAG_LABELS[feature],
                    "Parameter value": float(xv),
                    "Parameter within AI training range": True,
                    "Kuznetsov equation X50 (m)": formula_value,
                    "AI X50 (m)": float(ai_value),
                })

        st.markdown("### Formula and AI response by parameter")
        st.caption("Each graph varies one AI model parameter at a time across its observed training range. The other AI inputs stay at your entered settings. The equation line changes only where the Kuznetsov equation uses that parameter; it stays flat for factors absent from the equation.")
        for start in range(0, len(FRAG_FEATURES), 2):
            graph_cols = st.columns(2, gap="large")
            for col, feature in zip(graph_cols, FRAG_FEATURES[start:start + 2]):
                with col:
                    rows = [r for r in sensitivity_rows if r["Varied parameter"] == FRAG_LABELS[feature]]
                    graph = pd.DataFrame(rows)
                    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=135)
                    ax.plot(graph["Parameter value"], graph["Kuznetsov equation X50 (m)"], color=AMBER, linewidth=2.5, label="Kuznetsov equation")
                    ax.plot(graph["Parameter value"], graph["AI X50 (m)"], color=TEAL, linewidth=2.5, label="AI model")
                    style_axis(ax, f"X50 response · {FRAG_LABELS[feature]}", FRAG_LABELS[feature], "X50 (m)")
                    ax.legend(frameon=False, fontsize=8, loc="best")
                    render_fig(fig)

        report_row = {
            **comparison["inputs"],
            **{f"AI input · {FRAG_LABELS[k]}": v for k, v in comparison["features"].items()},
            "Kuznetsov equation X50 (m)": comparison["math"],
            "AI X50 (m)": comparison["ai"],
            "AI minus equation (m)": comparison["ai"] - comparison["math"],
            "AI training rows": frag_training_count,
            "AI training range warning": "; ".join(comparison["out_of_range"]),
        }
        if np.isfinite(comparison["inputs"]["Measured X50 (m)"]):
            measured = comparison["inputs"]["Measured X50 (m)"]
            report_row["Equation absolute error (m)"] = abs(comparison["math"] - measured)
            report_row["AI absolute error (m)"] = abs(comparison["ai"] - measured)
        st.download_button(
            "DOWNLOAD X50 CALCULATION REPORT · CSV",
            pd.DataFrame([report_row]).to_csv(index=False).encode("utf-8-sig"),
            "mibppm_x50_formula_vs_ai_report.csv",
            "text/csv",
            key="download_x50_report",
        )
        st.download_button(
            "DOWNLOAD FORMULA + AI CURVES · CSV",
            pd.DataFrame(sensitivity_rows).to_csv(index=False).encode("utf-8-sig"),
            "mibppm_x50_formula_vs_ai_parameter_curves.csv",
            "text/csv",
            key="download_x50_curves",
        )

# ---- Sensitivity: every input parameter for one selected output ----
with tab_sensitivity:
    st.markdown("### Parameter sensitivity")
    st.markdown(
        '<p class="small-note">Select an output to inspect how each of the five model inputs changes its estimate. '
        'Each panel varies one input while holding the other four constant.</p>',
        unsafe_allow_html=True
    )

    selected_target = st.selectbox(
        "Predicted output",
        TARGETS,
        key="sbpi_v3_sensitivity_target"
    )
    base = st.session_state.get("sbpi_v2_inputs", DEFAULTS)
    base = {f: float(base.get(f, DEFAULTS[f])) for f in FEATURES}

    st.markdown("**Reference input settings**")
    ref_cols = st.columns(5)
    for col, feature in zip(ref_cols, FEATURES):
        col.metric(feature, f"{base[feature]:.3f}")

    st.caption(
        "The shaded region represents the observed range in the research dataset. "
        "A straight response line is expected from the currently selected Linear Regression model; "
        "it is not evidence of a nonlinear physical relationship."
    )

    for row_start in range(0, len(FEATURES), 2):
        chart_cols = st.columns(2, gap="large")
        for col, feature in zip(chart_cols, FEATURES[row_start:row_start + 2]):
            with col:
                observed = numeric(data[feature])
                dmin, dmax = float(observed.min()), float(observed.max())
                pad = max((dmax - dmin) * 0.08, 0.01)
                lo, hi = dmin - pad, dmax + pad
                xgrid = np.linspace(lo, hi, 60)

                rows = []
                for xv in xgrid:
                    row = dict(base)
                    row[feature] = float(xv)
                    rows.append(row)
                curve_frame = pd.DataFrame(rows, columns=FEATURES)
                yp = np.asarray(
                    models[selected_target].predict(curve_frame),
                    dtype=float
                ).reshape(-1)

                fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=135)
                ax.axvspan(
                    dmin, dmax, color=TEAL, alpha=0.09,
                    label="Observed data range", zorder=0
                )
                ax.plot(
                    xgrid, yp, color=TEAL, linewidth=2.8,
                    label="Model response", zorder=3
                )

                current_x = base[feature]
                if lo <= current_x <= hi:
                    current_row = pd.DataFrame([base], columns=FEATURES)
                    current_y = float(
                        models[selected_target].predict(current_row)[0]
                    )
                    ax.scatter(
                        [current_x], [current_y], s=72, color=AMBER,
                        edgecolor="white", linewidth=1.3,
                        label="Reference setting", zorder=5
                    )
                    ax.axvline(
                        current_x, color=AMBER, alpha=0.45,
                        linestyle=":", linewidth=1.2
                    )

                style_axis(
                    ax,
                    f"{selected_target} vs {feature}",
                    feature,
                    f"Predicted {selected_target} ({UNITS[selected_target]})"
                )
                ax.legend(frameon=False, fontsize=8, loc="best")
                render_fig(fig)

    # Export the plotted response data so the analysis can be reused outside the dashboard.
    curve_rows = []
    for feature in FEATURES:
        observed = numeric(data[feature])
        dmin, dmax = float(observed.min()), float(observed.max())
        pad = max((dmax - dmin) * 0.08, 0.01)
        xgrid = np.linspace(dmin - pad, dmax + pad, 60)
        rows = []
        for xv in xgrid:
            row = dict(base)
            row[feature] = float(xv)
            rows.append(row)
        curve_frame = pd.DataFrame(rows, columns=FEATURES)
        predictions = np.asarray(models[selected_target].predict(curve_frame), dtype=float).reshape(-1)
        for xv, prediction in zip(xgrid, predictions):
            curve_rows.append({
                "Output": selected_target,
                "Output unit": UNITS[selected_target],
                "Varied parameter": feature,
                "Parameter value": float(xv),
                "Model estimate": float(prediction),
                "Reference setting": base[feature],
                "Within observed range": bool(dmin <= xv <= dmax),
                **{f"Held input · {name}": value for name, value in base.items() if name != feature},
            })
    curve_csv = pd.DataFrame(curve_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "DOWNLOAD ALL FIVE PARAMETER RESPONSE CURVES · CSV",
        curve_csv,
        f"mibppm_{selected_target.lower().replace(' ', '_').replace('/', '_')}_response_curves.csv",
        "text/csv",
        key="download_response_curves",
    )
    st.caption("Curve data covers each of the five inputs at 60 points, holding the other four at the reference settings. These are model responses, not equation outputs.")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("**Interpretation**")
    st.caption(
        "These are model sensitivity curves, not independent physical equations. "
        "They show the fitted model's response when one input changes and the others remain fixed. "
        "Because the saved SBPI models are Linear Regression pipelines, the curves will be straight lines."
    )

# ---- Validation diagnostics, separate per target ----
with tab_validation:
    st.caption("Validation uses grouped out-of-fold predictions from the supplied research dataset; results are exploratory and may not generalize to new mines.")
    st.markdown("### Grouped cross-validation diagnostics")
    st.markdown('<p class="small-note">Validation plots use saved out-of-fold predictions. They are not generated from the fitted model’s training predictions.</p>', unsafe_allow_html=True)
    metrics_path = RESULTS / "model_metrics.csv"
    oof_path = RESULTS / "oof_predictions.csv"
    best_path = RESULTS / "best_models.json"
    if metrics_path.exists():
        mt = pd.read_csv(metrics_path)
        # Select best recorded algorithm per target by MAE; preserves target-specific models.
        if {"Target", "Model", "MAE", "RMSE", "R2"}.issubset(mt.columns):
            best_metrics = mt.sort_values("MAE").groupby("Target", as_index=False).first()
            st.dataframe(best_metrics[["Target", "Model", "MAE", "RMSE", "R2"]], use_container_width=True, hide_index=True)
    else:
        st.info("model_metrics.csv was not found. Prediction still works; validation charts need this file and the OOF data.")
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        best_map = {}
        if best_path.exists():
            with open(best_path, encoding="utf-8") as fp:
                best_json = json.load(fp)
            best_map = {k: v.get("best_model") for k, v in best_json.items()}
        target = st.selectbox("Validation output", TARGETS, key="sbpi_v2_validation_target")
        if {"Target", "Model", "Actual", "Predicted"}.issubset(oof.columns):
            model_name = best_map.get(target)
            p = oof[(oof["Target"] == target) & ((oof["Model"] == model_name) if model_name else True)].copy()
            if not p.empty:
                p["Residual"] = p["Actual"] - p["Predicted"]
                mae = float(np.mean(np.abs(p["Residual"])))
                rmse = float(np.sqrt(np.mean(np.square(p["Residual"]))))
                m1, m2, m3 = st.columns(3)
                m1.metric("OOF records", len(p))
                m2.metric("MAE", f"{mae:.4f} {UNITS[target]}")
                m3.metric("RMSE", f"{rmse:.4f} {UNITS[target]}")
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
                    ax.scatter(p["Actual"], p["Predicted"], s=48, color=TEAL, alpha=.82, edgecolor="white", linewidth=.5)
                    mn = float(min(p["Actual"].min(), p["Predicted"].min()))
                    mx = float(max(p["Actual"].max(), p["Predicted"].max()))
                    pad = max((mx - mn) * .06, 1e-9)
                    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], linestyle="--", color=AMBER, linewidth=1.6, label="Ideal agreement")
                    style_axis(ax, f"{target} · measured vs predicted", f"Measured ({UNITS[target]})", f"Predicted ({UNITS[target]})")
                    ax.legend(frameon=False, fontsize=8)
                    render_fig(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
                    ax.scatter(p["Predicted"], p["Residual"], s=48, color=TEAL, alpha=.82, edgecolor="white", linewidth=.5)
                    ax.axhline(0, linestyle="--", color=AMBER, linewidth=1.6)
                    style_axis(ax, f"{target} · residual distribution", f"Predicted ({UNITS[target]})", f"Residual ({UNITS[target]})")
                    render_fig(fig)
                fig, ax = plt.subplots(figsize=(10, 3.7), dpi=120)
                ax.hist(p["Residual"], bins=min(12, max(5, len(p)//3)), color=TEAL, alpha=.82, edgecolor="white")
                style_axis(ax, f"{target} · residual frequency", f"Residual ({UNITS[target]})", "Record count")
                render_fig(fig)
            else:
                st.warning(f"No matching out-of-fold records were found for {target}.")
        else:
            st.warning("OOF file is missing required columns: Target, Model, Actual, Predicted.")
    else:
        st.info("oof_predictions.csv is required for out-of-fold validation charts.")

# ---- Data profile ----
with tab_data:
    st.markdown("### Dataset profile")
    st.caption("Observed values in the SBPI modelling dataset. Shaded graph ranges in the response tab refer to these observations.")
    profile = []
    for f in FEATURES + [t for t in TARGETS if t in data.columns]:
        s = numeric(data[f]) if f in data.columns else pd.Series(dtype=float)
        if len(s):
            profile.append({"Parameter": f, "Count": len(s), "Minimum": s.min(), "25th percentile": s.quantile(.25), "Median": s.median(), "75th percentile": s.quantile(.75), "Maximum": s.max(), "Std. deviation": s.std()})
    st.dataframe(pd.DataFrame(profile).round(4), use_container_width=True, hide_index=True)
    for f in FEATURES:
        s = numeric(data[f])
        fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
        ax.hist(s, bins=min(12, max(5, len(s)//3)), color=TEAL, alpha=.82, edgecolor="white")
        style_axis(ax, f"{f} · observed distribution", f, "Record count")
        render_fig(fig)

st.markdown('<div style="border-top:1px solid #DCE4E1;margin-top:25px;padding-top:13px;color:#7B898C;font-size:.74rem;">MIBPPM · SBPI research dashboard · Model estimates are not operational blast-design instructions.</div>', unsafe_allow_html=True)

