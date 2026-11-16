"""Data-analytics layer: lead-lag, NLP topics, segmentation, forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import PLATFORMS


def wide(interest: pd.DataFrame, topic: str) -> pd.DataFrame:
    return interest[interest.topic == topic].pivot(index="week", columns="platform", values="interest")[PLATFORMS]


# ---------- cross-platform lead-lag (time-series) ----------
def lead_lag(interest: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Lag (weeks) at which each platform best correlates with Instagram."""
    rows = []
    for topic in interest.topic.unique():
        w = wide(interest, topic)
        ig = w["Instagram"].values
        row = {"topic": topic}
        for p in PLATFORMS[1:]:
            other = w[p].values
            best = max(range(0, max_lag + 1), key=lambda k: np.corrcoef(ig[: len(ig) - k], other[k:])[0, 1])
            row[f"{p}_lag_weeks"] = best
            row[f"{p}_corr"] = round(float(np.corrcoef(ig[: len(ig) - best], other[best:])[0, 1]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- NLP topic modelling ----------
def topic_model(posts: pd.DataFrame, n_topics: int = 8, top_n: int = 6):
    vec = TfidfVectorizer(min_df=5, max_df=0.6)
    X = vec.fit_transform(posts.text)
    nmf = NMF(n_components=n_topics, random_state=0, init="nndsvda", max_iter=400)
    doc_topic = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    top = {f"Theme {i + 1}": list(terms[c.argsort()[::-1][:top_n]]) for i, c in enumerate(nmf.components_)}
    return top, doc_topic.argmax(axis=1)


# ---------- audience segmentation ----------
def segment(aud: pd.DataFrame, k: int = 4):
    X = StandardScaler().fit_transform(aud)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    out = aud.assign(segment=labels)
    profile = out.groupby("segment").mean().round(1)
    profile["size_pct"] = (out.segment.value_counts(normalize=True).sort_index() * 100).round(1)
    return out, profile


def name_segments(profile: pd.DataFrame) -> dict:
    names = {}
    for s, r in profile.iterrows():
        if r.news_hrs == profile.news_hrs.max():
            names[s] = "News-first mature viewers"
        elif r.ott_spend_inr == profile.ott_spend_inr.max():
            names[s] = "Premium OTT binge-watchers"
        elif r.instagram_hrs == profile.instagram_hrs.max():
            names[s] = "Gen-Z social discoverers"
        else:
            names[s] = "YouTube-led explorers"
    return names


# ---------- predictive modelling ----------
HORIZON = 4


def _features(w: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=w.index)
    f["ig"], f["yt"], f["news"], f["ott"] = w["Instagram"], w["YouTube"], w["News"], w["OTT"]
    f["ig_lag2"] = w["Instagram"].shift(2)
    f["ig_slope4"] = w["Instagram"].diff(4)
    f["yt_slope4"] = w["YouTube"].diff(4)
    f["ott_slope4"] = w["OTT"].diff(4)
    return f


def forecast_ott(interest: pd.DataFrame, test_weeks: int = 20):
    """Predict OTT interest HORIZON weeks ahead from upstream platforms.

    Returns (predictions for latest week, metrics dict). Chronological split, no leakage.
    """
    tr_X, tr_y, te_X, te_y, latest = [], [], [], [], {}
    for topic in interest.topic.unique():
        w = wide(interest, topic)
        f = _features(w)
        y = w["OTT"].shift(-HORIZON)
        ok = f.notna().all(axis=1) & y.notna()
        X, yy = f[ok], y[ok]
        cut = len(X) - test_weeks
        tr_X.append(X.iloc[:cut]); tr_y.append(yy.iloc[:cut])
        te_X.append(X.iloc[cut:]); te_y.append(yy.iloc[cut:])
        latest[topic] = f.dropna().iloc[[-1]]
    tr_X, tr_y, te_X, te_y = map(pd.concat, (tr_X, tr_y, te_X, te_y))
    model = Ridge(alpha=1.0).fit(tr_X, tr_y)
    pred = model.predict(te_X)
    metrics = {
        "model_MAE": float(np.mean(np.abs(pred - te_y))),
        "naive_MAE": float(np.mean(np.abs(te_X["ott"] - te_y))),
    }
    metrics["improvement_pct"] = 100 * (1 - metrics["model_MAE"] / metrics["naive_MAE"])
    preds = {t: float(model.predict(x)[0]) for t, x in latest.items()}
    return preds, metrics


# ---------- business layer: recommendation engine ----------
def recommend(interest: pd.DataFrame, preds: dict, audience_per_point: int = 8000,
              arpu_inr: float = 90, cost_cr: float = 2.5) -> pd.DataFrame:
    """Turn forecasts into invest / pilot / avoid decisions with value estimates (INR crore)."""
    rows = []
    for topic, p in preds.items():
        w = wide(interest, topic)
        cur = float(w["OTT"].iloc[-1])
        ig = w["Instagram"]
        momentum = float(ig.iloc[-1] - ig.iloc[-5])
        stage = "Emerging" if momentum > 3 else ("Peaking" if ig.iloc[-1] > 0.8 * ig.max() else "Fading/Dormant")
        viewers = max(p, 0) * audience_per_point
        value_cr = viewers * arpu_inr * 3 / 1e7  # ~3 months of subscription value
        roi = (value_cr - cost_cr) / cost_cr
        action = "INVEST NOW" if roi > 0.25 and p > cur else ("PILOT / WATCH" if roi > 0 else "AVOID")
        rows.append({"topic": topic, "stage": stage, "ott_now": round(cur, 1), "ott_forecast_4w": round(p, 1),
                     "est_viewers": int(viewers), "est_value_cr": round(value_cr, 2),
                     "roi": round(roi, 2), "action": action})
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)
