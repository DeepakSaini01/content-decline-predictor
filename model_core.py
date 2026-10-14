"""Core logic for the content-decline prototype (no Streamlit imports, so it is testable)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEED = 42
# trend_pct and the 30-day window columns are what trend_direction is computed from -> label leakage
LEAK = ["trend_pct", "trend_direction",
        "impressions_last_30d", "clicks_last_30d", "sessions_last_30d",
        "impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d"]
IDS = ["content_id", "client_id"]
DROP = LEAK + IDS + ["y", "provider_used"]   # provider_used is ~71% missing


def prepare(df: pd.DataFrame):
    """Return X (features), y (1 = declining), groups (client), feature lists."""
    df = df.copy()
    df["y"] = (df["trend_direction"].str.lower() == "down").astype(int)
    feats = [c for c in df.columns if c not in DROP]
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feats if c not in cat_cols]
    X = df[feats].copy()
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str).astype("category")
    return X, df["y"], df["client_id"], num_cols, cat_cols


def new_model():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        categorical_features="from_dtype", early_stopping=True, random_state=SEED)


def _pr(s):  # percentile rank in [0, 1]
    return (s.rank(method="average") - 1) / (len(s) - 1)


def baseline_score(d: pd.DataFrame) -> pd.Series:
    """Week-4 deterministic refresh score (same weights as scripts/02_baseline_score.py)."""
    vis = _pr(np.log1p(d["impressions_90d"]))
    fresh = _pr(d["days_since_last_update"])
    pos = d["avg_position"].clip(1, 50)
    pos_opp = (1 - (pos - pos.min()) / (pos.max() - pos.min())) * vis * (d["avg_position"] > 0)
    depth = (1 - _pr(d["word_count"].fillna(0))) * vis
    return (0.40 * vis + 0.30 * fresh + 0.25 * pos_opp + 0.05 * depth).clip(0, 1)


def precision_at_k(y_true, scores, k):
    idx = np.argsort(np.asarray(scores))[::-1][:k]
    return float(np.asarray(y_true)[idx].mean())


def metrics(y_true, scores, ks=(50, 200)):
    out = {"ROC-AUC": roc_auc_score(y_true, scores),
           "Avg precision": average_precision_score(y_true, scores)}
    for k in ks:
        out[f"Precision@{k}"] = precision_at_k(y_true, scores, k)
    return out


def evaluate(df, X, y, groups, n_folds=5):
    """Compare baseline vs model on a random split and on unseen-client folds."""
    from sklearn.model_selection import train_test_split
    rows = []
    # random stratified 70/30 (the notebook's split)
    tr, va = train_test_split(X.index, test_size=0.3, stratify=y, random_state=SEED)
    m = new_model().fit(X.loc[tr], y[tr])
    rows.append({"split": "Random 70/30", "model": "Week-4 baseline",
                 **metrics(y[va], baseline_score(df.loc[va]))})
    rows.append({"split": "Random 70/30", "model": "Gradient boosting",
                 **metrics(y[va], m.predict_proba(X.loc[va])[:, 1])})
    # client-grouped: test clients never seen in training
    gss = GroupShuffleSplit(n_splits=n_folds, test_size=0.3, random_state=SEED)
    g = []
    for tr_i, va_i in gss.split(X, y, groups):
        tr, va = X.index[tr_i], X.index[va_i]
        m = new_model().fit(X.loc[tr], y[tr])
        g.append({"model": "Week-4 baseline", **metrics(y[va], baseline_score(df.loc[va]))})
        g.append({"model": "Gradient boosting", **metrics(y[va], m.predict_proba(X.loc[va])[:, 1])})
    g = pd.DataFrame(g).groupby("model").agg(["mean", "std"])
    return pd.DataFrame(rows), g


def out_of_fold_risk(X, y, groups, n_splits=5):
    """Risk score for every row from a model that never saw that row's client."""
    oof = pd.Series(np.nan, index=X.index)
    for tr_i, va_i in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = new_model().fit(X.iloc[tr_i], y.iloc[tr_i])
        oof.iloc[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    return oof


def holdout_importance(X, y, groups, n_repeats=3, max_rows=5000):
    """Permutation importance (drop in ROC-AUC) on unseen clients."""
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(X, y, groups))
    m = new_model().fit(X.iloc[tr_i], y.iloc[tr_i])
    va_i = np.random.RandomState(SEED).permutation(va_i)[:max_rows]   # subsample: much faster
    r = permutation_importance(m, X.iloc[va_i], y.iloc[va_i], n_repeats=n_repeats,
                               random_state=SEED, scoring="roc_auc", n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def default_row(X, num_cols, cat_cols):
    """Typical content: median of numerics, most common category."""
    row = {c: float(X[c].median()) for c in num_cols}
    row.update({c: X[c].mode().iloc[0] for c in cat_cols})
    return row


def predict_one(model, X_ref, row: dict) -> float:
    r = pd.DataFrame([row])[X_ref.columns]
    for c in X_ref.select_dtypes("category").columns:
        r[c] = pd.Categorical(r[c], categories=X_ref[c].cat.categories)
    return float(model.predict_proba(r)[:, 1][0])
