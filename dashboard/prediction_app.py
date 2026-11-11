
from pathlib import Path
import sys
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Project paths and CAOS package
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CAOS_ROOT = ROOT.parent / "CAOS_BlastFrag-main"

if str(CAOS_ROOT) not in sys.path:
    sys.path.insert(0, str(CAOS_ROOT))

from blastfrag.types import Pattern
from blastfrag.classical import (
    kuznetsov_x50_m,
    cunningham_uniformity_index,
    rosin_rammler,
)
from blastfrag.rockfactor import SITE_ROCK_FACTOR


DATA_PATH = ROOT / "data" / "raw" / "caos_surface_confirmed.csv"

FEATURES = [
    "S_over_B",
    "H_over_B",
    "B_over_D",
    "T_over_B",
    "Pf_kg_m3",
    "XB_m",
    "E_GPa",
]

SITE_NAMES = [
    "Akdaglar",
    "Dongri-Buzurg",
    "Enusa",
    "Mrica",
    "Murgul",
    "Ozmert",
    "Reocin",
    "Soma",
]

# ---------------------------------------------------------
# Load and train AI model
# ---------------------------------------------------------

@st.cache_resource
def load_ai_model():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=FEATURES + ["x50_m"])

    model = LinearRegression()

    model.fit(
        df[FEATURES],
        df["x50_m"],
    )

    feature_ranges = {
        feature: (
            float(df[feature].min()),
            float(df[feature].max()),
        )
        for feature in FEATURES
    }

    return model, feature_ranges, len(df)


model, feature_ranges, training_count = load_ai_model()

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="MIBPPM | Blast Prediction",
    page_icon="⛏️",
    layout="wide",
)

st.title("MIBPPM — Mathematical & AI Blast Prediction")
st.caption(
    "Surface-mining fragmentation research prototype | Experiment 6"
)

st.info(
    "Research decision-support prototype. Predictions are estimates, "
    "not independently validated blast-design recommendations."
)

# ---------------------------------------------------------
# Sidebar: site selection
# ---------------------------------------------------------

st.sidebar.header("Mine information")

site_options = ["New / Unlisted Mine"] + SITE_NAMES

selected_site = st.sidebar.selectbox(
    "Select mining site",
    site_options,
)

if selected_site == "New / Unlisted Mine":

    rock_factor = st.sidebar.number_input(
        "Documented rock factor (A)",
        min_value=0.1,
        max_value=30.0,
        value=8.0,
        step=0.1,
        help="Enter a justified rock factor. Do not use the default without verification.",
    )

    st.sidebar.warning(
        "New mine: the rock factor must be supplied and independently justified."
    )

else:

    rock_factor = float(SITE_ROCK_FACTOR[selected_site])

    st.sidebar.metric(
        "2012-derived site rock factor",
        f"{rock_factor:.2f}",
    )

    st.sidebar.caption(
        "This factor was back-solved from published predictions. "
        "It is not an independently measured rock factor."
    )

# ---------------------------------------------------------
# Input parameters
# ---------------------------------------------------------

st.header("1. Blast parameters")

left, right = st.columns(2)

with left:

    st.subheader("Blast geometry")

    burden = st.number_input(
        "Burden (m)",
        min_value=0.1,
        max_value=30.0,
        value=3.0,
        step=0.1,
    )

    spacing = st.number_input(
        "Spacing (m)",
        min_value=0.1,
        max_value=40.0,
        value=3.0,
        step=0.1,
    )

    bench_height = st.number_input(
        "Bench height (m)",
        min_value=0.5,
        max_value=60.0,
        value=12.0,
        step=0.5,
    )

    hole_diameter = st.number_input(
        "Hole diameter (mm)",
        min_value=25.0,
        max_value=500.0,
        value=100.0,
        step=5.0,
    )

    stemming = st.number_input(
        "Stemming length (m)",
        min_value=0.0,
        max_value=30.0,
        value=2.0,
        step=0.1,
    )

    subdrill = st.number_input(
        "Subdrill (m)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
    )

with right:

    st.subheader("Rock and explosive properties")

    powder_factor = st.number_input(
        "Powder factor (kg/m³)",
        min_value=0.01,
        max_value=5.0,
        value=0.8,
        step=0.05,
    )

    rock_modulus = st.number_input(
        "Young's modulus (GPa)",
        min_value=0.1,
        max_value=200.0,
        value=60.0,
        step=1.0,
    )

    in_situ_block = st.number_input(
        "In-situ block size XB (m)",
        min_value=0.001,
        max_value=10.0,
        value=0.3,
        step=0.01,
    )

    st.caption(
        "Explosive assumption: ANFO, RWS = 100. "
        "This matches the CAOS training dataset."
    )

# ---------------------------------------------------------
# Feature construction
# ---------------------------------------------------------

feature_values = {
    "S_over_B": spacing / burden,
    "H_over_B": bench_height / burden,
    "B_over_D": burden / (hole_diameter / 1000.0),
    "T_over_B": stemming / burden,
    "Pf_kg_m3": powder_factor,
    "XB_m": in_situ_block,
    "E_GPa": rock_modulus,
}

input_df = pd.DataFrame(
    [feature_values],
    columns=FEATURES,
)

# ---------------------------------------------------------
# Range warnings
# ---------------------------------------------------------

out_of_range = []

for feature in FEATURES:

    low, high = feature_ranges[feature]
    value = feature_values[feature]

    if value < low or value > high:

        out_of_range.append(
            f"{feature}: supplied {value:.4f}, "
            f"training range {low:.4f} to {high:.4f}"
        )

if out_of_range:

    st.warning(
        "Some inputs are outside the training-data range. "
        "AI predictions may be unreliable."
    )

    for warning in out_of_range:
        st.write("- " + warning)

else:

    st.success(
        "All seven AI input features are within their observed "
        "training-data ranges. This does not guarantee accuracy."
    )

# ---------------------------------------------------------
# Optional actual measurement
# ---------------------------------------------------------

st.header("2. Optional actual measurement")

actual_x50 = st.number_input(
    "Measured X50 (m), if available",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.01,
    help="Enter 0 if no measured X50 is available.",
)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.header("3. Calculate predictions")

if st.button(
    "Run mathematical + AI prediction",
    type="primary",
    use_container_width=True,
):

    try:

        if burden <= 0 or spacing <= 0 or bench_height <= 0:
            raise ValueError("Geometry values must be positive.")

        if stemming >= bench_height + subdrill:
            raise ValueError(
                "Stemming must be shorter than bench height plus subdrill."
            )

        if rock_factor <= 0:
            raise ValueError("Rock factor must be positive.")

        # Build the CAOS pattern.
        pattern = Pattern(
            burden_m=burden,
            spacing_m=spacing,
            bench_height_m=bench_height,
            hole_diameter_mm=hole_diameter,
            stemming_m=stemming,
            powder_factor_kg_m3=powder_factor,
            subdrill_m=subdrill,
        )

        # Mathematical prediction.
        math_x50 = kuznetsov_x50_m(
            pattern,
            rock_factor,
        )

        # AI prediction.
        ai_x50 = float(
            model.predict(input_df)[0]
        )

        # The current experimental 50:50 ensemble.
        hybrid_x50 = 0.5 * math_x50 + 0.5 * ai_x50

        # Physical and numerical validity checks.
        if not np.isfinite(math_x50) or math_x50 <= 0:
            raise ValueError(
                "Mathematical prediction is not physically valid."
            )

        if not np.isfinite(ai_x50) or ai_x50 <= 0:
            raise ValueError(
                "AI prediction is non-positive or invalid. "
                "Do not use the hybrid result."
            )

        # Cunningham uniformity and Rosin-Rammler distribution.
        uniformity = cunningham_uniformity_index(pattern)

        distribution = rosin_rammler(
            hybrid_x50,
            uniformity,
        )

        # Store results for report.
        st.session_state["prediction_result"] = {
            "site": selected_site,
            "rock_factor": rock_factor,
            "math_x50": math_x50,
            "ai_x50": ai_x50,
            "hybrid_x50": hybrid_x50,
            "uniformity": uniformity,
            "distribution": distribution,
            "feature_values": feature_values,
            "actual_x50": actual_x50 if actual_x50 > 0 else None,
            "out_of_range": out_of_range,
            "pattern": pattern,
        }

    except Exception as error:

        st.error(f"Prediction could not be completed: {error}")

# ---------------------------------------------------------
# Display stored results
# ---------------------------------------------------------

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.header("4. Prediction results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mathematical X50",
        f"{result['math_x50']:.4f} m",
    )

    c2.metric(
        "AI X50",
        f"{result['ai_x50']:.4f} m",
    )

    c3.metric(
        "Hybrid X50",
        f"{result['hybrid_x50']:.4f} m",
    )

    st.caption(
        "Hybrid prediction = arithmetic mean of mathematical "
        "and AI predictions. This fixed 50:50 method was the "
        "best aggregate method in Experiment 5."
    )

    st.metric(
        "Cunningham uniformity index",
        f"{result['uniformity']:.4f}",
    )

    # Actual vs predicted comparison.
    if result["actual_x50"] is not None:

        actual = result["actual_x50"]

        st.subheader("Measured versus predicted")

        comparison = pd.DataFrame({
            "Method": [
                "Mathematical",
                "AI",
                "Hybrid",
            ],
            "Predicted X50 (m)": [
                result["math_x50"],
                result["ai_x50"],
                result["hybrid_x50"],
            ],
        })

        comparison["Measured X50 (m)"] = actual

        comparison["Absolute error (m)"] = abs(
            comparison["Predicted X50 (m)"] - actual
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Fragmentation distribution graph
    # -----------------------------------------------------

    st.header("5. Predicted fragmentation distribution")

    distribution = result["distribution"]

    sizes = np.asarray(distribution.sizes_m)
    passing = np.asarray(distribution.passing) * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        sizes,
        passing,
        label="Rosin–Rammler predicted distribution",
    )

    ax.axvline(
        result["hybrid_x50"],
        linestyle="--",
        label=f"Hybrid X50 = {result['hybrid_x50']:.3f} m",
    )

    ax.axhline(
        50,
        linestyle=":",
        label="50% passing",
    )

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Fragment size (m)")
    ax.set_ylabel("Cumulative passing (%)")
    ax.set_title("Predicted Fragmentation Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # Downloadable CSV report
    # -----------------------------------------------------

    st.header("6. Export prediction report")

    report = pd.DataFrame([
        {
            "site": result["site"],
            "rock_factor": result["rock_factor"],
            "mathematical_x50_m": result["math_x50"],
            "ai_x50_m": result["ai_x50"],
            "hybrid_x50_m": result["hybrid_x50"],
            "uniformity": result["uniformity"],
            "actual_x50_m": result["actual_x50"],
            **result["feature_values"],
        }
    ])

    st.download_button(
        "Download prediction CSV",
        data=report.to_csv(index=False),
        file_name="mibppm_prediction_report.csv",
        mime="text/csv",
    )

    curve_report = pd.DataFrame({
        "fragment_size_m": sizes,
        "cumulative_passing_percent": passing,
    })

    st.download_button(
        "Download fragmentation curve CSV",
        data=curve_report.to_csv(index=False),
        file_name="mibppm_fragmentation_curve.csv",
        mime="text/csv",
    )

    st.caption(
        "The fragmentation curve is a mathematical distribution "
        "derived from predicted X50 and Cunningham uniformity. "
        "It is not a measured particle-size distribution."
    )

# ---------------------------------------------------------
# Research information
# ---------------------------------------------------------

with st.expander("Research methodology and limitations"):

    st.write(
        """
        **Mathematical model**

        Kuznetsov X50 using the CAOS_BlastFrag implementation.
        Site factors are back-solved from published predictions.

        **AI model**

        Linear Regression trained on all 85 selected surface records.
        The model is intended for research exploration. Its training
        predictions are not independent validation results.

        **Hybrid model**

        Fixed 50:50 average of mathematical and AI X50 estimates.

        **Fragmentation distribution**

        Rosin-Rammler distribution using Cunningham uniformity.

        **Limitations**

        - 85 records from eight sites.
        - Limited geological and explosive coverage.
        - Site factors are calibrated rather than independently measured.
        - New-site predictions require independent rock-factor justification.
        - No independent external validation has been performed.
        - The distribution curve is estimated, not measured.
        """
    )

st.divider()

st.caption(
    f"MIBPPM Research Prototype | AI training records: {training_count} | "
    "Not for operational blast-design approval."
)
