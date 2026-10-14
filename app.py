import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Content Decline Predictor", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.joblib"

if not MODEL_PATH.exists():
    st.error("model.joblib not found. Upload it to the Space next to app.py.")
    st.stop()


@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    bundle["model"].n_jobs = 1
    return bundle


b = load_bundle()
rf, FEATURES = b["model"], b["features"]
IMPORTANCES = pd.Series(b["importances"]).sort_values(ascending=False)


@st.cache_data(show_spinner="Scoring file...")
def score_file(file_bytes: bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes))
    missing = [c for c in FEATURES if c not in raw.columns]
    subset = raw.reindex(columns=FEATURES)
    Xb = subset.apply(pd.to_numeric, errors="coerce")
    bad = int(Xb.isna().sum().sum() - subset.isna().sum().sum())
    Xb = Xb.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    out = raw.copy()
    out["decline_probability"] = rf.predict_proba(Xb)[:, 1]
    out = out.sort_values("decline_probability", ascending=False).reset_index(drop=True)
    out.insert(0, "risk_rank", np.arange(1, len(out) + 1))
    return out, missing, bad


@st.cache_data(show_spinner=False)
def scored_csv(file_bytes: bytes, threshold: float):
    out, _, _ = score_file(file_bytes)
    out = out.copy()
    out["predicted_declining"] = (out["decline_probability"] >= threshold).astype(int)
    return out.to_csv(index=False).encode()


def show_df(data, **kwargs):
    st.dataframe(data, use_container_width=True, **kwargs)


def precision_at_k(y_true, scores, k):
    y_arr = np.asarray(y_true)
    k = min(int(k), len(y_arr))
    idx = np.argsort(np.asarray(scores))[::-1][:k]
    return float(y_arr[idx].mean())


st.title("Content Decline Predictor")
st.caption("Random Forest · decision-support tool · scores are directional estimates, not guarantees.")

if b.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {b.get('sklearn_version')} but this app runs "
        f"{sklearn.__version__}. Pin the same version in requirements.txt."
    )

# ---------- Sidebar ----------
with st.sidebar:
    p = b["params"]
    st.header("Model card")
    st.write(f"RandomForest · {p['n_estimators']} trees · min_samples_leaf={p['min_samples_leaf']} "
             f"· max_depth={p['max_depth']}")
    st.write(f"{len(FEATURES)} numeric features · stratified 70/30 split")
    st.write(f"Train rows: {b['n_train']:,} · Validation rows: {b['n_val']:,}")
    st.metric("Validation ROC-AUC", f"{b['val_auc']:.3f}",
              f"baseline {b['baseline_auc']:.3f}", delta_color="off")
    st.metric("Validation Precision@50", f"{b['val_prec50']:.2f}",
              f"baseline {b['baseline_prec50']:.2f}", delta_color="off")
    st.caption("Precision@50 is measured on only 50 rows, so treat it as directional.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption("Leakage columns (trend_pct and the 30-day windows) were excluded from training.")

view = st.radio(
    "View",
    ["Single page", "Batch CSV", "What drives the model", "Validation results"],
    horizontal=True,
    label_visibility="collapsed",
)

# ---------- Single prediction ----------
if view == "Single page":
    st.subheader("Enter the signals for one page")
    with st.form("single"):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                values[f] = st.number_input(
                    f, value=float(b["defaults"][f]), format="%.4f", key=f"in_{f}"
                )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([values])[FEATURES].astype("float32")
        prob = float(rf.predict_proba(row)[0, 1])
        c1, c2 = st.columns(2)
        c1.metric("Probability of declining", f"{prob:.1%}")
        c2.metric("Prediction", "Declining" if prob >= threshold else "Not declining")
        st.progress(min(max(prob, 0.0), 1.0))

# ---------- Batch scoring ----------
elif view == "Batch CSV":
    st.subheader("Score many pages and rank them")
    st.write("Upload a CSV using the training column names. Missing columns or values are "
             "filled with 0, exactly as in training. Extra columns are ignored by the model.")
    file = st.file_uploader("CSV file", type="csv")
    top_k = int(st.number_input("Top-K to highlight", 1, 1000, 50))

    if file is not None:
        file_bytes = file.getvalue()
        try:
            out, missing, bad = score_file(file_bytes)
        except Exception as e:
            st.error(f"Could not read or score the CSV: {e}")
            st.stop()

        if out.empty:
            st.warning("The uploaded file has no rows.")
            st.stop()
        if len(missing) == len(FEATURES):
            st.error("None of the model's feature columns were found in this file.")
            st.stop()
        if missing:
            st.warning(f"{len(missing)} feature column(s) missing, filled with 0: {missing}")
        if bad > 0:
            st.warning(f"{bad} non-numeric value(s) were found and replaced with 0.")

        flagged = int((out["decline_probability"] >= threshold).sum())
        st.write(f"Rows scored: {len(out):,} · flagged declining at threshold "
                 f"{threshold:.2f}: {flagged:,}")

        if "trend_direction" in out.columns:
            y_true = (out["trend_direction"].astype(str).str.lower().str.strip() == "down").astype(int)
            if y_true.nunique() == 2:
                k = min(top_k, len(out))
                st.info(
                    f"Labels found. ROC-AUC on this file: "
                    f"{roc_auc_score(y_true, out['decline_probability']):.3f} · "
                    f"Precision@{k}: {precision_at_k(y_true, out['decline_probability'], k):.2f}. "
                    "If this file overlaps the training data, these numbers are optimistic."
                )

        st.write(f"Top {top_k} pages by decline probability:")
        top = out.head(top_k).copy()
        top["predicted_declining"] = (top["decline_probability"] >= threshold).astype(int)
        show_df(top)

        st.download_button("Download all scores (CSV)",
                           scored_csv(file_bytes, float(threshold)),
                           "scored_pages.csv", "text/csv")

# ---------- Importance ----------
elif view == "What drives the model":
    st.subheader("Feature importance (impurity-based)")
    st.bar_chart(IMPORTANCES.head(15))
    st.caption("Shows which signals the model relies on. It is not proof of cause.")

# ---------- Validation ----------
else:
    st.subheader("Validation results (from training run)")
    show_df(
        pd.DataFrame({
            "Metric": ["ROC-AUC", "Precision@50"],
            "Baseline": [b["baseline_auc"], b["baseline_prec50"]],
            "Random Forest": [b["val_auc"], b["val_prec50"]],
        }),
        hide_index=True,
    )
    cm = b["confusion_matrix"]
    st.write("Confusion matrix at threshold 0.5")
    show_df(pd.DataFrame(
        cm,
        index=["Actual 0 (not declining)", "Actual 1 (declining)"],
        columns=["Pred 0", "Pred 1"],
    ))
    acc = (cm[0][0] + cm[1][1]) / b["n_val"]
    st.caption(f"Accuracy at 0.5 threshold: {acc:.1%}. ROC-AUC and the confusion matrix "
               "are steadier evidence than Precision@50.")
