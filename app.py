"""Content-decline early-warning prototype.  Run:  streamlit run app.py"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

import model_core as mc

st.set_page_config(page_title="Content decline prototype", page_icon="📉", layout="wide")
st.title("📉 Content decline early-warning (prototype)")
st.caption("Observed, directional decision-support: which content is more likely to be declining "
           "(trend down ≥ 20%), using only signals that do not leak the label.")

# ---------------------------------------------------------------- data
DEFAULTS = [Path("content_refresh_anonymized.csv"), Path("data/content_refresh_anonymized.csv"),
            Path("../content_refresh_anonymized.csv"),
            Path("/mnt/user-data/uploads/content_refresh_anonymized.csv")]
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("content_refresh_anonymized.csv", type="csv")
    st.caption("Needs the `trend_direction`, `client_id`, `content_id` columns.")


@st.cache_data(show_spinner="Reading data…")
def load(file_bytes, path):
    return pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else pd.read_csv(path)


path = next((p for p in DEFAULTS if p.exists()), None)
if up is None and path is None:
    st.info("Upload the CSV in the sidebar to begin.")
    st.stop()
df = load(up.getvalue() if up else None, str(path) if path else None)
need = {"trend_direction", "client_id", "content_id", "impressions_90d",
        "days_since_last_update", "avg_position", "word_count"}
if not need.issubset(df.columns):
    st.error(f"Missing columns: {sorted(need - set(df.columns))}")
    st.stop()
X, y, groups, num_cols, cat_cols = mc.prepare(df)


# ---------------------------------------------------------------- cached model work
@st.cache_resource(show_spinner="Training final model…")
def final_model(_X, _y, key):
    return mc.new_model().fit(_X, _y)


@st.cache_data(show_spinner="Evaluating on random and unseen-client splits…")
def cached_eval(key):
    return mc.evaluate(df, X, y, groups)


@st.cache_data(show_spinner="Scoring every row with a client-blind model…")
def cached_oof(key):
    return mc.out_of_fold_risk(X, y, groups)


@st.cache_data(show_spinner="Computing feature importance…")
def cached_imp(key):
    return mc.holdout_importance(X, y, groups)


key = (len(df), float(df["impressions_90d"].sum()))
model = final_model(X, y, key)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Clients", df["client_id"].nunique())
c3.metric("Declining share", f"{y.mean():.1%}")
c4.metric("Features used", f"{len(num_cols)} num + {len(cat_cols)} cat")

tab_rank, tab_one, tab_perf, tab_imp = st.tabs(
    ["🔎 Rank content", "🧪 What-if", "📊 How good is it?", "🧭 Drivers"])

# ---------------------------------------------------------------- tab 1: ranking
with tab_rank:
    st.subheader("Highest decline risk")
    st.caption("Scores come from a model that never saw that row's client (out-of-fold), "
               "so they are an honest preview of how it behaves on new clients.")
    oof = cached_oof(key)
    scored = df[["content_id", "client_id", "content_type", "avg_position", "impressions_90d",
                 "days_since_last_update", "content_age_days"]].copy()
    scored["decline_risk"] = oof.round(3)
    scored["week4_baseline"] = mc.baseline_score(df).round(3)
    scored["actually_declining"] = y.map({1: "yes", 0: "no"})
    a, b, c = st.columns(3)
    top_n = a.slider("Show top N", 10, 500, 50, 10)
    ctypes = ["(all)"] + sorted(scored["content_type"].dropna().unique())
    ctype = b.selectbox("Content type", ctypes)
    min_imp = c.number_input("Min impressions (90d)", 0, int(df["impressions_90d"].max()), 0)
    view = scored[scored["impressions_90d"] >= min_imp]
    if ctype != "(all)":
        view = view[view["content_type"] == ctype]
    view = view.sort_values("decline_risk", ascending=False).head(top_n)
    st.dataframe(view, use_container_width=True, hide_index=True)
    if len(view):
        base_top = scored.sort_values("week4_baseline", ascending=False).head(top_n)
        st.write(f"Of these {len(view)}, **{(view.actually_declining == 'yes').mean():.0%}** were "
                 f"actually declining (observed). Week-4 baseline's top {top_n}: "
                 f"**{(base_top.actually_declining == 'yes').mean():.0%}**. "
                 f"Overall base rate: {y.mean():.0%}.")
    st.download_button("Download this list (CSV)", view.to_csv(index=False).encode(),
                       "decline_risk_top.csv", "text/csv")

# ---------------------------------------------------------------- tab 2: what-if
with tab_one:
    st.subheader("Score one piece of content")
    st.caption("Unset fields use the dataset median / most common value.")
    base = mc.default_row(X, num_cols, cat_cols)
    key_num = ["avg_position", "ctr", "impressions_90d", "clicks_90d", "days_with_impressions",
               "content_age_days", "days_since_last_update", "word_count"]
    key_num = [k for k in key_num if k in num_cols]
    row = dict(base)
    cols = st.columns(4)
    for i, k in enumerate(key_num):
        lo, hi = float(X[k].min()), float(X[k].quantile(0.99))
        row[k] = cols[i % 4].number_input(k, lo, float(X[k].max()), float(base[k]),
                                          step=max((hi - lo) / 100, 0.01), key=f"n_{k}")
    cat_show = [c for c in ["content_type", "main_intent", "competition_level"] if c in cat_cols]
    cols = st.columns(max(len(cat_show), 1))
    for i, k in enumerate(cat_show):
        opts = list(X[k].cat.categories)
        row[k] = cols[i].selectbox(k, opts, index=opts.index(base[k]))
    p = mc.predict_one(model, X, row)
    st.metric("Estimated decline risk", f"{p:.0%}", delta=f"{p - y.mean():+.0%} vs. average",
              delta_color="inverse")
    st.progress(min(max(p, 0.0), 1.0))

# ---------------------------------------------------------------- tab 3: performance
with tab_perf:
    st.subheader("Model vs. Week-4 baseline")
    rand, grp = cached_eval(key)
    st.markdown("**Random 70/30 split** (what the notebook used). Optimistic: the same client "
                "appears in both train and test.")
    st.dataframe(rand.drop(columns="split").round(3), hide_index=True, use_container_width=True)
    st.markdown("**Unseen-client split** (mean over 5 folds). This is the number to quote for new clients.")
    st.dataframe(grp.round(3), use_container_width=True)
    st.info("Precision@K varies a lot between folds (see the std columns), so treat it as directional. "
            "Only 32 clients, one holding ~23% of the rows, and each client's declining share ranges "
            "from 0% to ~94%.")

# ---------------------------------------------------------------- tab 4: drivers
with tab_imp:
    st.subheader("Which signals matter most")
    st.caption("Permutation importance on unseen clients: drop in ROC-AUC when a feature is shuffled. "
               "Importance shows association, not cause.")
    imp = cached_imp(key)
    top = imp.head(15)
    st.bar_chart(top.set_index("feature")["importance"], horizontal=True)
    st.dataframe(top.round(4), hide_index=True, use_container_width=True)
    st.warning("`impressions_90d` and `days_with_impressions` cover a window that includes the last "
               "30 days, so they may carry some of the decline signal indirectly.")
