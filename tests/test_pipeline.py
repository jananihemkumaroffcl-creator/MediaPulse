from src.data import generate_interest, generate_posts, generate_audience
from src import analysis as an


def test_lead_lag_orders_platforms():
    ll = an.lead_lag(generate_interest())
    assert ll.YouTube_lag_weeks.median() < ll.News_lag_weeks.median() < ll.OTT_lag_weeks.median()


def test_forecast_beats_naive():
    _, m = an.forecast_ott(generate_interest())
    assert m["model_MAE"] < m["naive_MAE"]


def test_recommend_segments_topics():
    i = generate_interest()
    preds, _ = an.forecast_ott(i)
    assert len(an.recommend(i, preds)) == 8
    _, prof = an.segment(generate_audience())
    assert len(prof) == 4
    top, _ = an.topic_model(generate_posts())
    assert len(top) == 8
