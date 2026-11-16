# 📊 Media Intelligence Dashboard

> Tracks how audience interest **flows across Instagram → YouTube → News → OTT** and turns that signal into data-backed content investment decisions.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  What It Does

Media companies struggle to know *which* topics to invest in before they peak. This dashboard solves that by:

- Detecting **cross-platform lead–lag patterns** (e.g. Instagram buzz precedes OTT demand by ~4 weeks)
- Classifying topics by **lifecycle stage**: `emerging` / `peaking` / `fading`
- Generating **investment recommendations**: `INVEST NOW` / `PILOT` / `AVOID`
- Forecasting **4-week-ahead OTT demand** using Ridge regression
- Segmenting audiences into **named personas** via K-Means clustering

---

## Architecture

```
Instagram ──┐
YouTube  ──┤──► Cross-correlation  ──► Topic Lifecycle  ──► Recommendations
News     ──┤    (lead–lag analysis)     (NMF + TF-IDF)      (ROI in ₹ crore)
OTT      ──┘
                                    ──► K-Means Personas
                                    ──► Ridge Forecast (4-week)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | Synthetic generator with configurable platform lags |
| NLP | TF-IDF + NMF topic modelling |
| ML | K-Means clustering, Ridge regression |
| Validation | Chronological train/test split vs. naive baseline |
| Dashboard | Streamlit |
| Testing | pytest |

---

## Quickstart

```bash
git clone https://github.com/<your-username>/media-intelligence-dashboard.git
cd media-intelligence-dashboard
python -m venv .venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

Run tests:
```bash
pytest
```

---

## Project Structure

```
├── app.py                  # Streamlit dashboard entry point
├── src/
│   ├── data.py             # Synthetic data generator (platform lags: 0/2/4/6 weeks)
│   └── analysis.py         # Cross-correlation, NLP, clustering, forecasting
├── tests/
│   └── test_pipeline.py    # Pipeline unit tests
└── requirements.txt
```

---

##  Key Design Decisions

- **Synthetic data with realistic lags** — makes the pipeline connector-ready; swap `src/data.py` generators with Google Trends / YouTube Data API / news feeds
- **Chronological validation** — forecast model is validated in time order against a naive baseline to prevent data leakage
- **Modular `src/` layout** — analysis logic is fully decoupled from the UI

---

## Roadmap

- [ ] Real API connectors (Google Trends, YouTube Data API)
- [ ] Prophet / LSTM forecasts
- [ ] BERTopic for richer topic modelling
- [ ] Granger causality tests for lead–lag significance
