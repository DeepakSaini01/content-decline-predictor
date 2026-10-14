"""Content-decline early-warning prototype.  Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

import model_core as mc

st.set_page_config(page_title="Content decline prototype", page_icon="📉", layout="wide")
st.title("📉 Content decline early-warning (prototype)")
st.caption("Observed, directional decision-support: which content is more likely to be declining "
           "(trend down ≥ 20%), using only signals that do not leak the label.")

# ------------------------------------------------------------------ data (local folder)
CSV = "content_refresh_anonymized.csv"
path = next((p for p in (Path(CSV), Path("data") / CSV, Path("data/raw") / CSV,
                         Path("../data/raw") / CSV, Path("..") / CSV) if p.exists()), None)
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
@st.cache_resource(show_spinner="Training final model…")
def final_model(p: str, mtime: float):
    return mc.new_model().fit(X, y)


@st.cache_data(show_spinner="Evaluating on random and unseen-client splits…")
def cached_eval(p: str, mtime: float):
    return mc.evaluate(df, X, y, groups)


@st.cache_data(show_spinner="Scoring every row with a client-blind model…")
def cached_oof(p: str, mtime: float):
    return mc.out_of_fold_risk(X, y, groups)


@st.cache_data(show_spinner="Computing feature importance…")
def cached_imp(p: str, mtime: float):
    return mc.holdout_importance(X, y, groups)


ck = (str(path), path.stat().st_mtime)
model = final_model(*ck)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Clients", groups.nunique())
c3.metric("Declining share", f"{y.mean():.1%}")
c4.metric("Features used", f"{len(num_cols)} num + {len(cat_cols)} cat")

# Only the selected section runs, so heavy work (CV, importance) happens when you open it.
section = st.radio("View", ["🔎 Rank content", "🧪 What-if", "📊 How good is it?", "🧭 Drivers"],
                   horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------------ rank content
if section.startswith("🔎"):
    st.subheader("Highest decline risk")
    st.caption("Scores come from a model that never saw that row's client (out-of-fold), "
               "so they preview how it behaves on new clients.")
    scored = df[["content_id", "client_id", "content_type", "avg_position", "impressions_90d",
                 "days_since_last_update", "content_age_days"]].assign(
        decline_risk=cached_oof(*ck).round(3),
        week4_baseline=base_score.round(3),
        actually_declining=y.map({1: "yes", 0: "no"}))
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
        st.write(f"Of these {len(view)}, **{(view.actually_declining == 'yes').mean():.0%}** were actually "
                 f"declining (observed). Week-4 baseline's top {top_n}: "
                 f"**{(base_top.actually_declining == 'yes').mean():.0%}**. "
                 f"Overall base rate: {y.mean():.0%}.")
    st.download_button("Download this list (CSV)", view.to_csv(index=False).encode(),
                       "decline_risk_top.csv", "text/csv")

# ------------------------------------------------------------------ what-if
elif section.startswith("🧪"):
    st.subheader("Score one piece of content")
    st.caption("Unset fields use the dataset median / most common value.")
    base = mc.default_row(X, num_cols, cat_cols)
    row = dict(base)
    key_num = [k for k in ["avg_position", "ctr", "impressions_90d", "clicks_90d", "days_with_impressions",
                           "content_age_days", "days_since_last_update", "word_count"] if k in num_cols]
    cols = st.columns(4)
    for i, k in enumerate(key_num):
        lo, hi = float(X[k].min()), float(X[k].max())
        row[k] = cols[i % 4].number_input(k, lo, hi, min(max(base[k], lo), hi),
                                          step=max((hi - lo) / 200, 0.01), key=f"n_{k}")
    cat_show = [c for c in ["content_type", "main_intent", "competition_level"] if c in cat_cols]
    cols = st.columns(max(len(cat_show), 1))
    for i, k in enumerate(cat_show):
        opts = list(X[k].cat.categories)
        row[k] = cols[i].selectbox(k, opts, index=opts.index(base[k]))
    p = mc.predict_one(model, X, row)
    st.metric("Estimated decline risk", f"{p:.0%}", delta=f"{p - y.mean():+.0%} vs. average",
              delta_color="inverse")
    st.progress(min(max(p, 0.0), 1.0))

# ------------------------------------------------------------------ performance
elif section.startswith("📊"):
    st.subheader("Model vs. Week-4 baseline")
    rand, grp = cached_eval(*ck)
    st.markdown("**Random 70/30 split** (as in the notebook). Optimistic: the same client appears "
                "in both train and test.")
    st.dataframe(rand.drop(columns="split").round(3), hide_index=True, use_container_width=True)
    st.markdown("**Unseen-client split** (mean/std over 5 folds). Quote this one for new clients.")
    st.dataframe(grp.round(3), use_container_width=True)
    st.info("Precision@K varies a lot between folds (see std), so treat it as directional. Only "
            "32 clients, one holding ~23% of the rows; each client's declining share ranges 0%–~94%.")

# ------------------------------------------------------------------ drivers
else:
    st.subheader("Which signals matter most")
    st.caption("Permutation importance on unseen clients (drop in ROC-AUC when a feature is shuffled). "
               "Shows association, not cause.")
    top = cached_imp(*ck).head(15)
    st.bar_chart(top.set_index("feature")["importance"], horizontal=True)
    st.dataframe(top.round(4), hide_index=True, use_container_width=True)
    st.warning("`impressions_90d` and `days_with_impressions` cover a window that includes the last "
               "30 days, so they may carry some decline signal indirectly.")
