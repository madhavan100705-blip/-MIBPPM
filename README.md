# MIBPPM — Hybrid Mathematical & AI Blasting Performance Research Prototype

A research-oriented starter application for **surface bench-blasting performance analysis**.
It keeps three values separate:
1. mathematical/empirical model output,
2. AI prediction trained on measured historical outcomes,
3. actual field measurement (ground truth).

> **Important:** This repository is a research prototype, not a blast-design or operational-control tool.
> It does not recommend explosive quantities, blast patterns, firing sequences, or safety distances.
> Any operational interpretation requires qualified mining engineers and applicable regulations.

## What is included
- Python calculation library for:
  - Kuznetsov mean-fragment-size equation (requires a supplied rock factor and correctly defined inputs),
  - Rosin–Rammler passing fraction,
  - scaled-distance PPV equation (requires site-calibrated K and beta),
  - volumetric and mass powder factor.
- CSV ingestion and schema checks.
- Separate ML training/evaluation per measured target using Random Forest and optional XGBoost.
- MAE, RMSE, R², bias and actual-vs-predicted outputs.
- FastAPI backend.
- Streamlit research dashboard.
- A clearly labelled synthetic demo dataset (not field data).

## Project layout
```text
MIBPPM/
├── backend/
│   ├── app/main.py
│   ├── app/schemas.py
│   ├── app/data.py
│   ├── app/models.py
│   └── app/engine/
│       ├── fragmentation.py
│       ├── vibration.py
│       └── powder_factor.py
├── dashboard/app.py
├── data/raw/
├── data/processed/
├── models/
├── scripts/
├── tests/
├── requirements.txt
└── .env.example
```

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```
Open `http://127.0.0.1:8000/docs`.

In a second terminal:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

## Public dataset acquisition\nSee `research/PUBLIC_DATASET_REGISTER.md` for verified research leads, dataset sizes, access status, and acquisition rules. Dataset descriptions in papers are not assumed to mean raw rows are openly downloadable.\n\n## Dataset format
Use one row per **blast event**. Keep source metadata and actual measured outputs. See
`data/README_DATA.md` for the column dictionary and example. Do not fill unknown actual outcomes
with theoretical estimates. Leave them blank/NaN.

## Train an AI model
```powershell
python scripts/train_models.py --csv data/raw/your_blast_data.csv --target PPV_actual
```
Only rows with a measured target are used for that target. The split is grouped by `mine_id` when
there are at least two mines; otherwise it uses a reproducible random split. For publication-grade
validation, replace this with a pre-registered temporal or leave-one-mine-out strategy and report
its limitations.

## Current scientific limitations
- Kuznetsov implementation is a formula calculator, not a calibrated universal predictor. Check
  the original equation's unit conventions and rock-factor definition before use.
- The PPV scaled-distance model requires site-calibrated constants K and beta. No defaults are
  invented.
- The extended PPV, flyrock, backbreak, air-overpressure, cost and productivity equations are not
  fabricated. Their coefficients/formulations must be selected from verified publications and
  calibrated against measured data before activation.
- Powder factor is calculated from recorded charge and rock quantity; it is not treated as an
  AI target by default.
- Synthetic demo data is only for testing the software pipeline and must never be reported as
  observed mine data or used to claim model accuracy.

## Research workflow
1. Acquire public source files and record citation, license, mine, measurement method and units.
2. Preserve originals in `data/raw/`; create cleaned copies in `data/processed/`.
3. Audit missingness, duplicates, units, and blast-event identity.
4. Implement and unit-test each published equation with source-specific assumptions.
5. Train models only against actual measured outputs.
6. Compare mathematical and AI outputs against the same held-out actual observations.
7. Report per-output sample counts, source/mine coverage, MAE, RMSE, R², bias and limitations.

## Safety and use
This prototype is for academic analysis. It must not be used to design, approve, or execute a blast.
Predictions of flyrock, vibration or air overpressure are not a substitute for statutory assessment,
instrumentation, exclusion-zone procedures, or competent engineering review.
