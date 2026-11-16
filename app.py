import streamlit as st
import plotly.express as px
from src.data import generate_interest, generate_posts, generate_audience, PLATFORMS
from src import analysis as an

st.set_page_config(page_title="Media Intelligence Dashboard", layout="wide")
st.title("📡 Media Intelligence Dashboard")
st.caption("How audience interest moves Instagram → YouTube → News → OTT, and where a media company should invest.")


@st.cache_data
def load():
    return generate_interest(), generate_posts(), generate_audience()


interest, posts, aud = load()

with st.sidebar:
    st.header("Business assumptions")
    per_pt = st.slider("Viewers per OTT interest point", 2000, 20000, 8000, 500)
    arpu = st.slider("ARPU (₹/month)", 30, 300, 90, 10)
    cost = st.slider("Content cost per title (₹ crore)", 0.5, 10.0, 2.5, 0.5)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trends", "Lead–Lag", "Topics (NLP)", "Audience", "Recommendations"])

with tab1:
    topic = st.selectbox("Topic", interest.topic.unique())
    d = interest[interest.topic == topic]
    st.plotly_chart(px.line(d, x="week", y="interest", color="platform", title=f"{topic}: weekly interest by platform"),
                    use_container_width=True)

with tab2:
    ll = an.lead_lag(interest)
    st.dataframe(ll, use_container_width=True)
    st.info(f"Median lag vs Instagram — YouTube: {ll.YouTube_lag_weeks.median():.0f}w, "
            f"News: {ll.News_lag_weeks.median():.0f}w, OTT: {ll.OTT_lag_weeks.median():.0f}w. "
            "Instagram is the early-warning signal.")

with tab3:
    top, assign = an.topic_model(posts)
    for k, v in top.items():
        st.write(f"**{k}**: " + ", ".join(v))

with tab4:
    seg, prof = an.segment(aud)
    names = an.name_segments(prof)
    prof.insert(0, "name", prof.index.map(names))
    st.dataframe(prof, use_container_width=True)
    st.plotly_chart(px.scatter(seg.sample(800, random_state=0), x="instagram_hrs", y="ott_hrs",
                               color=seg.loc[seg.sample(800, random_state=0).index, "segment"].astype(str),
                               title="Segments: social vs OTT hours"), use_container_width=True)

with tab5:
    preds, m = an.forecast_ott(interest)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model MAE", f"{m['model_MAE']:.2f}")
    c2.metric("Naive MAE", f"{m['naive_MAE']:.2f}")
    c3.metric("Improvement", f"{m['improvement_pct']:.0f}%")
    rec = an.recommend(interest, preds, per_pt, arpu, cost)
    st.dataframe(rec, use_container_width=True)
    st.plotly_chart(px.bar(rec, x="topic", y="est_value_cr", color="action", title="Estimated 3-month value (₹ crore)"),
                    use_container_width=True)
