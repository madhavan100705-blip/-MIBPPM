import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from backend.app.data import dataset_summary
from backend.app.engine.powder_factor import powder_factor_mass, powder_factor_volume
from backend.app.engine.fragmentation import rosin_rammler_passing_fraction

st.set_page_config(page_title="MIBPPM Research Dashboard", layout="wide")
st.title("MIBPPM | Hybrid Mathematical & AI Research Dashboard")
st.caption("Surface bench-blasting research prototype — not an operational blast-design tool.")

uploaded = st.file_uploader("Upload a blast-event CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Dataset overview")
    st.write(dataset_summary(df) if {"blast_id","mine_id","source_id"}.issubset(df.columns)
             else {"rows": len(df), "columns": list(df.columns),
                   "warning": "Add blast_id, mine_id and source_id for provenance-aware analysis."})
    st.dataframe(df.head(100), use_container_width=True)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        col = st.selectbox("Choose a measured output for distribution inspection", numeric)
        st.plotly_chart(px.histogram(df, x=col, title=f"Distribution: {col}"), use_container_width=True)
else:
    st.info("Upload a public research dataset in CSV format to begin. Use the data dictionary in data/README_DATA.md.")

st.divider()
st.subheader("Transparent mathematical calculators")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Powder factor (mass basis)**")
    q = st.number_input("Explosive mass (kg)", min_value=0.0, value=100.0)
    m = st.number_input("Rock mass (tonnes)", min_value=0.001, value=1000.0)
    st.metric("Powder factor (kg/t)", f"{powder_factor_mass(q,m):.5f}")
with c2:
    st.markdown("**Rosin–Rammler passing fraction**")
    x = st.number_input("Particle size x", min_value=0.0, value=50.0)
    xc = st.number_input("Characteristic size xc", min_value=0.001, value=60.0)
    n = st.number_input("Uniformity index n", min_value=0.001, value=1.5)
    st.metric("Fraction passing", f"{rosin_rammler_passing_fraction(x,xc,n):.5f}")
