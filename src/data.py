"""Synthetic cross-platform data generator.

Real social/OTT APIs are gated, so we simulate a realistic *lead-lag* structure:
a topic starts on Instagram, reaches YouTube ~2 weeks later, news ~4 weeks later
and OTT ~6 weeks later. Swap these functions for real connectors later.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PLATFORMS = ["Instagram", "YouTube", "News", "OTT"]
LAG_WEEKS = {"Instagram": 0, "YouTube": 2, "News": 4, "OTT": 6}
AMPLITUDE = {"Instagram": 1.0, "YouTube": 0.85, "News": 0.7, "OTT": 0.6}

TOPICS = {
    "Space Documentaries": ["rocket", "isro", "moon", "orbit", "launch", "mission", "galaxy"],
    "True Crime": ["case", "investigation", "murder", "detective", "trial", "mystery", "evidence"],
    "Regional Cinema": ["tamil", "malayalam", "telugu", "cinema", "director", "film", "boxoffice"],
    "Cricket Docuseries": ["cricket", "ipl", "captain", "stadium", "worldcup", "legend", "innings"],
    "Wellness & Yoga": ["yoga", "meditation", "wellness", "mindful", "ayurveda", "fitness", "retreat"],
    "Startup Stories": ["startup", "founder", "funding", "unicorn", "pitch", "growth", "venture"],
    "Food Heritage": ["recipe", "streetfood", "chef", "heritage", "spice", "kitchen", "tradition"],
    "Climate Action": ["climate", "solar", "emissions", "flood", "green", "policy", "renewable"],
}
GENERIC = ["watch", "new", "viral", "latest", "love", "india", "best", "story", "today", "must"]


def generate_interest(seed: int = 42, weeks: int = 104) -> pd.DataFrame:
    """Weekly interest index (0-100) per topic x platform."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=weeks, freq="W-MON")
    t = np.arange(weeks)
    frames = []
    for i, topic in enumerate(TOPICS):
        # two topics are deliberately 'emerging' now so the dashboard has live opportunities
        start = weeks + 1 + i // 2 if i in (0, 3) else int(rng.integers(10, weeks - 30))
        peak = rng.uniform(55, 95)
        width = rng.uniform(4, 8)
        for p in PLATFORMS:
            base = rng.uniform(8, 20)
            sig = base + AMPLITUDE[p] * peak * np.exp(-0.5 * ((t - (start + LAG_WEEKS[p])) / width) ** 2)
            sig = np.clip(sig + rng.normal(0, 2.5, weeks), 0, 100)
            frames.append(pd.DataFrame({"week": dates, "topic": topic, "platform": p, "interest": sig.round(2)}))
    return pd.concat(frames, ignore_index=True)


def generate_posts(seed: int = 7, n: int = 3000) -> pd.DataFrame:
    """Short synthetic posts/headlines for NLP topic modelling."""
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    rows = []
    for _ in range(n):
        topic = topics[rng.integers(len(topics))]
        words = list(rng.choice(TOPICS[topic], size=4)) + list(rng.choice(GENERIC, size=3))
        rng.shuffle(words)
        rows.append({"platform": PLATFORMS[rng.integers(len(PLATFORMS))], "true_topic": topic, "text": " ".join(words)})
    return pd.DataFrame(rows)


def generate_audience(seed: int = 11, n: int = 2500) -> pd.DataFrame:
    """Synthetic viewer panel with four latent segments."""
    rng = np.random.default_rng(seed)
    seg = rng.choice(4, size=n, p=[0.3, 0.25, 0.25, 0.2])
    prof = {  # age, ig, yt, news, ott hours/wk, ott spend INR/mo
        0: (22, 9, 6, 1, 4, 150),
        1: (31, 3, 5, 3, 10, 450),
        2: (45, 1, 2, 8, 6, 300),
        3: (27, 5, 9, 2, 7, 250),
    }
    rows = []
    for s in seg:
        a, ig, yt, nw, ott, sp = prof[s]
        rows.append([max(16, rng.normal(a, 4)), max(0, rng.normal(ig, 1.5)), max(0, rng.normal(yt, 1.5)),
                     max(0, rng.normal(nw, 1)), max(0, rng.normal(ott, 1.5)), max(0, rng.normal(sp, 60))])
    return pd.DataFrame(rows, columns=["age", "instagram_hrs", "youtube_hrs", "news_hrs", "ott_hrs", "ott_spend_inr"])
