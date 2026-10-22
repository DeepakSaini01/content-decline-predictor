"""Content-decline early-warning prototype.  Run:  streamlit run app.py

Aligned with the ML-08 Capstone notebook (w05_model.ipynb):
- Algorithm: RandomForestClassifier (200 trees, min_samples_leaf=2)
- Features: 23 numeric signals
- Split: stratified random 70/30 (random_state=42)
- Metrics: ROC-AUC, Precision@50
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import model_core as mc

st.set_page_config(page_title="Content decline prototype", page_icon="📉", layout="wide")
st.title("📉 Content decline early-warning (prototype)")
st.caption(
    "Observed, directional decision-support: which content is more likely to be declining "
    "(trend down ≥ 20%), using only non-leaking signals. "
    "Model: RandomForestClassifier (aligned with ML-08 notebook)."
)

# ------------------------------------------------------------------ data (local folder)
CSV = "content_refresh_anonymized.csv"
path = next(
    (
        p
        for p in (
            Path(CSV),
            Path("data") / CSV,
            Path("data/raw") / CSV,
            Path("../data/raw") / CSV,
            Path("..") / CSV,
        )
        if p.exists()
    ),
    None,
)
if path is None:
    st.error(f"Could not find {CSV}. Put it next to app.py or in data/ or data/raw/.")
    st.stop()
st.sidebar.success(f"Data: {path}")


@st.cache_data(show_spinner="Reading data…")
def load(p: str, mtime: float):
    """mtime in the signature => cache refreshes if the file changes."""
    df = pd.read_csv(p)
    X, y, groups, num_cols, cat_cols = mc.prepare(df)
    return df, X, y, groups, num_cols, cat_cols, mc.baseline_score(df)


df, X, y, groups, num_cols, cat_cols, base_score = load(str(path), path.stat().st_mtime)


# ------------------------------------------------------------------ cached model work
@st.cache_resource(show_spinner="Training Random Forest model…")
def final_model(p: str, mtime: float):
    return mc.new_model().fit(X, y)


@st.cache_data(show_spinner="Evaluating on stratified 70/30 split…")
def cached_eval(p: str, mtime: float):
    return mc.evaluate(df, X, y, groups)


@st.cache_data(show_spinner="Scoring all items…")
def cached_oof(p: str, mtime: float):
    return mc.out_of_fold_risk(X, y, groups)


ck = (str(path), path.stat().st_mtime)
model = final_model(*ck)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Clients", groups.nunique())
c3.metric("Declining share", f"{y.mean():.1%}")
c4.metric("Features used", f"{len(num_cols)} numeric")

section = st.radio(
    "View",
    ["🔎 Rank content", "🧪 What-if", "📊 How good is it?"],
    horizontal=True,
    label_visibility="collapsed",
)

# ------------------------------------------------------------------ rank content
if section.startswith("🔎"):
    st.subheader("Highest decline risk")
    st.caption("Scores generated from the Random Forest model (stratified 70/30 split).")
    scored = df[[
        "content_id",
        "client_id",
        "content_type",
        "avg_position",
        "impressions_90d",
        "days_since_last_update",
        "content_age_days",
    ]].assign(
        decline_risk=cached_oof(*ck).round(3),
        week4_baseline=base_score.round(3),
        actually_declining=y.map({1: "yes", 0: "no"}),
    )
    a, b, c = st.columns(3)
    top_n = a.slider("Show top N", 10, 500, 50, 10)
    ctype = b.selectbox("Content type", ["(all)"] + sorted(scored["content_type"].dropna().unique()))
    min_imp = c.number_input("Min impressions (90d)", 0, int(df["impressions_90d"].max()), 0)

    view = scored[scored["impressions_90d"] >= min_imp]
    if ctype != "(all)":
        view = view[view["content_type"] == ctype]
    view = view.nlargest(top_n, "decline_risk")
    st.dataframe(view, use_container_width=True, hide_index=True)
    if len(view):
        base_top = scored.nlargest(top_n, "week4_baseline")
        st.write(
            f"Of these {len(view)}, **{(view.actually_declining == 'yes').mean():.0%}** were actually "
            f"declining (observed). Week-4 baseline's top {top_n}: "
            f"**{(base_top.actually_declining == 'yes').mean():.0%}**. "
            f"Overall base rate: {y.mean():.0%}."
        )
    st.download_button(
        "Download this list (CSV)",
        view.to_csv(index=False).encode(),
        "decline_risk_top.csv",
        "text/csv",
    )

# ------------------------------------------------------------------ what-if
elif section.startswith("🧪"):
    st.subheader("Score one piece of content")
    st.caption("Adjust key numeric features. Unset features use dataset median values.")
    base = mc.default_row(X, num_cols, cat_cols)
    row = dict(base)
    key_num = [
        k
        for k in [
            "avg_position",
            "ctr",
            "impressions_90d",
            "clicks_90d",
            "days_with_impressions",
            "content_age_days",
            "days_since_last_update",
            "word_count",
        ]
        if k in num_cols
    ]
    cols = st.columns(4)
    for i, k in enumerate(key_num):
        lo, hi = float(X[k].min()), float(X[k].max())
        row[k] = cols[i % 4].number_input(
            k,
            lo,
            hi,
            min(max(base[k], lo), hi),
            step=max((hi - lo) / 200, 0.01),
            key=f"n_{k}",
        )
    p = mc.predict_one(model, X, row)
    st.metric(
        "Estimated decline risk",
        f"{p:.0%}",
        delta=f"{p - y.mean():+.0%} vs. average",
        delta_color="inverse",
    )
    st.progress(min(max(p, 0.0), 1.0))

# ------------------------------------------------------------------ performance
else:
    section.startswith("📊"):
    st.subheader("Model vs. Week-4 baseline")
    results = cached_eval(*ck)
    st.markdown(
        "**Stratified 70/30 split** (matching `w05_model.ipynb` with `random_state=42`)."
    )
    st.dataframe(results.round(3), hide_index=True, use_container_width=True)
    st.info(
        "Recreates the notebook metrics: Random Forest achieves **ROC-AUC ~0.767** and "
        "**Precision@50 = 98.0%** (vs. Baseline ROC-AUC 0.580 and Precision@50 44.0%)."
    )

