# 📡 Media Intelligence Dashboard

Identifies how audience interest **moves across Instagram → YouTube → News → OTT** and turns that into content-investment recommendations for a media company.

## Business analytics
- Market opportunity detection (topic lifecycle stage: emerging / peaking / fading)
- Content investment recommendations (INVEST NOW / PILOT / AVOID)
- Audience segmentation with named personas
- Estimated business value & ROI (₹ crore, editable assumptions)
- Strategic recommendation table

## Data analytics
- Cross-platform lead–lag (cross-correlation) analysis
- NLP topic modelling (TF-IDF + NMF)
- Audience segmentation (K-Means)
- Time-series features + 4-week-ahead OTT forecast (Ridge), validated chronologically against a naive baseline
- Interactive Streamlit dashboard

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
pytest
```

## Notes
Data is **synthetic** (`src/data.py`) with a built-in 0/2/4/6-week platform lag. The pipeline is connector-ready: replace the generators with Google Trends / YouTube Data API / news / OTT catalogue data.

## Roadmap
Real API connectors · Prophet/LSTM forecasts · BERTopic · Granger causality tests
