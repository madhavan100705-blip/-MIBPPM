from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------------------
# MIBPPM | SBPI research dashboard — clean editorial / technical visual system
# Uses existing local SBPI dataset, saved pipelines, grouped OOF predictions.
# Does not modify training data, models, or experiment outputs.
# -----------------------------------------------------------------------------
st.set_page_config(page_title="MIBPPM | Research Console V3", page_icon="⛏", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sbpi"
MODEL_DIR = RESULTS / "saved_models"

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
.block-container {{ max-width:1560px; padding-top:1.3rem; padding-bottom:3.2rem; }}
h1,h2,h3 {{ font-family:'Manrope',sans-serif !important; color:{INK} !important; letter-spacing:-.035em; }}
h1 {{ font-size:2.05rem !important; font-weight:800 !important; }}
h2 {{ font-size:1.35rem !important; font-weight:750 !important; }}
h3 {{ font-size:1.04rem !important; font-weight:700 !important; }}
p, label {{ color:{INK}; }}
[data-testid="stSidebar"] {{ background:#202D33; }}
[data-testid="stSidebar"] * {{ color:#EDF3F1 !important; }}
[data-testid="stSidebar"] [data-testid="stMetric"] {{ background:#2A3A40; border:1px solid #41545A; }}
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
except Exception as exc:
    st.error(f"Dashboard data/model loading failed: {exc}")
    st.info("Confirm that this file is inside the project's dashboard folder and that results/sbpi contains the dataset and saved_models folder.")
    st.stop()

missing = [f for f in FEATURES if f not in data.columns]
if missing:
    st.error("Required model input columns are missing: " + ", ".join(missing))
    st.stop()

# Sidebar metadata
with st.sidebar:
    st.markdown("# MIBPPM")
    st.caption("BENCH BLASTING · RESEARCH CONSOLE")
    st.divider()
    st.markdown("**Study dataset**")
    st.metric("Records", f"{len(data):,}")
    if "Blast ID" in data.columns:
        st.metric("Blast groups", f"{data['Blast ID'].nunique():,}")
    st.metric("Model outputs", str(len(TARGETS)))
    st.divider()
    st.caption("SBPI model set")
    st.caption("Linear regression pipelines · Grouped 5-fold validation")
    st.caption("Prediction scope: exploratory estimates from the supplied research dataset.")

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">MIBPPM / BENCH BLAST PERFORMANCE</div>
  <h1>Research analysis console</h1>
  <div class="hero-sub">Blast-condition inputs, model estimates, parameter response curves and grouped validation diagnostics.</div>
</div>
""", unsafe_allow_html=True)

# Shared summary
rcols = st.columns(4)
for col, label, val in zip(rcols, ["DATA RECORDS", "BLAST GROUPS", "MODEL INPUTS", "PREDICTED OUTPUTS"], [len(data), data["Blast ID"].nunique() if "Blast ID" in data else "—", len(FEATURES), len(TARGETS)]):
    col.markdown(f'<div class="output-card"><div class="kicker">{label}</div><div class="output-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
tab_predict, tab_sensitivity, tab_validation, tab_data = st.tabs(["PREDICTION", "PARAMETER RESPONSE", "MODEL VALIDATION", "DATA PROFILE"])

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
        estimates = st.session_state.get("sbpi_v2_estimates")
        if estimates:
            cards = st.columns(2)
            for i, target in enumerate(TARGETS):
                with cards[i % 2]:
                    st.markdown(f'<div class="output-card"><div class="output-name">{target}</div><div class="output-value">{estimates[target]:,.4f}</div><div class="output-unit">{UNITS[target]} · AI model estimate</div></div>', unsafe_allow_html=True)
                    st.write("")
            csv = pd.DataFrame([{"Output": t, "AI prediction": estimates[t], "Unit": UNITS[t]} for t in TARGETS]).to_csv(index=False).encode("utf-8")
            st.download_button("EXPORT PREDICTIONS · CSV", csv, "mibppm_sbpi_predictions.csv", "text/csv")
        else:
            st.markdown('<div class="output-card"><div class="kicker">MODEL OUTPUTS</div><div class="output-value">—</div><p class="small-note">Run prediction to populate the five output estimates.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("### Equation-based outputs")
    st.caption("The existing SBPI input table does not contain the full input set and calibration constants needed to calculate these independently. Unavailable equation values are intentionally left blank rather than estimated or invented.")
    math_status = pd.DataFrame([
        ["P80 (mm)", "Not configured", "Bench height, charge mass, rock factor, RWS and uniformity inputs required"],
        ["PPV (mm/s)", "Not configured", "Charge per delay, monitoring distance and site constants K, β required"],
        ["Flyrock (m)", "Not configured", "Stemming, charge and selected site-calibrated equation required"],
        ["Crusher Productivity (t/h)", "Not configured", "Crusher capacity and operating-condition inputs required"],
        ["SBPI", "Not configured", "Original index equation, weights and normalization required"],
    ], columns=["Output", "Equation result", "Missing equation inputs"])
    st.dataframe(math_status, use_container_width=True, hide_index=True)

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

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("**Interpretation**")
    st.caption(
        "These are model sensitivity curves, not independent physical equations. "
        "They show the fitted model's response when one input changes and the others remain fixed. "
        "Because the saved SBPI models are Linear Regression pipelines, the curves will be straight lines."
    )

# ---- Validation diagnostics, separate per target ----
with tab_validation:
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
    st.warning("The SBPI dataset is small. High grouped-validation scores are exploratory and do not establish performance at new mines or operating conditions.")

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
