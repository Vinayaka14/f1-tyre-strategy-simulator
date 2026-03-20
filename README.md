# 🏎 F1 Tyre Strategy Simulator

A real-time pit stop decision engine for 
Formula 1 races, powered by XGBoost and a 
rule-based strategy system.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](STREAMLIT_URL_PLACEHOLDER)

## What It Does

This simulator analyses F1 race telemetry 
and recommends whether a driver should pit 
or stay out, using two complementary systems:

- **Rule Engine** — 11 priority-ordered rules 
  encoding F1 strategic principles (undercut 
  windows, safety car opportunities, critical 
  degradation thresholds)
- **XGBoost Classifier** — trained on 65,001 
  laps across 3 seasons (2022–2024) with 
  ROC-AUC of 0.9950

## Features

### Real Race Replay Mode
- Select any race from 2023–2024 seasons
- Scrub lap-by-lap through actual race data
- Live event overrides: Safety Car, VSC, 
  sudden rain, competitor pit stops
- Position bar showing all 20 drivers with 
  team colours and tyre compounds
- Tyre degradation simulation vs actual 
  race lap times
- What-if scenario history log with CSV export

### Manual Scenario Builder
- Build any hypothetical race scenario
- Independent race/circuit selector
- Lap time projection, undercut simulation,
  optimal pit window, compound comparison,
  stint outcome analysis
- Fully reactive — updates on every change

## Tech Stack

| Component | Technology |
|---|---|
| Data source | FastF1 API (official F1 timing) |
| ML model | XGBoost (ROC-AUC: 0.9950) |
| Training data | 65,001 laps, 2022–2024 |
| Comparison | XGBoost vs LightGBM vs MLP |
| Frontend | Streamlit + Plotly |
| Deployment | Streamlit Community Cloud |

## Model Performance

| Model | F1 Score | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| XGBoost ✅ | 0.836 | 0.995 | 0.748 | 0.948 |
| LightGBM | 0.832 | 0.995 | 0.738 | 0.954 |
| MLP | 0.722 | 0.987 | 0.587 | 0.936 |

## Project Structure

```
f1-tyre-strategy-simulator/
├── app.py                    # Streamlit app
├── train.py                  # Model training
├── compare_models.py         # Model comparison
├── src/
│   ├── data_loader.py        # FastF1 data fetch
│   ├── feature_eng.py        # Feature engineering
│   ├── tyre_model.py         # Degradation curves
│   ├── rule_engine.py        # 11-rule strategy engine
│   ├── ml_model.py           # XGBoost classifier
│   ├── strategy.py           # Decision combiner
│   └── models/
│       ├── lgbm_model.py     # LightGBM
│       ├── mlp_model.py      # Neural network
│       └── xgb_model.py      # XGBoost (comparison)
├── models/
│   ├── best_classifier.pkl   # Deployed model
│   └── threshold.json        # Optimal threshold
├── data/
│   └── processed/
│       └── features.csv      # Engineered features
├── docs/
│   └── model_comparison.md   # Model evaluation report
└── .streamlit/
    └── config.toml           # Dark F1 theme
```

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/f1-tyre-strategy-simulator
cd f1-tyre-strategy-simulator
pip install -r requirements.txt
streamlit run app.py
```

Note: First load of a race session fetches 
from FastF1 and may take 30–60 seconds. 
Subsequent loads use local cache.

## Data Sources

- **FastF1** — official F1 timing and telemetry
- **Seasons covered** — 2022, 2023, 2024
- **Training examples** — 65,001 labelled laps
- **Features** — 12 engineered features including 
  tyre age, degradation rate, gap deltas, 
  fuel-adjusted lap time, safety car proxy

## About

Built as a portfolio project demonstrating 
end-to-end ML engineering in a motorsport 
context — data pipeline, feature engineering, 
model training and comparison, rule engine 
design, and production deployment.

---
*Data provided by FastF1. Not affiliated 
with Formula 1 or the FIA.*
