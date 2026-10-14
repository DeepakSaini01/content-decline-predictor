"""Core logic for the content-decline prototype (no Streamlit imports, so it is testable).

Aligned with the ML-08 Capstone notebook (w05_model.ipynb):
- Algorithm: RandomForestClassifier (n_estimators=200, min_samples_leaf=2, random_state=42)
- Features: numeric only (23 features), fillna(0)
- Split: stratified random 70/30 (random_state=42)
- Metrics: ROC-AUC, Precision@50
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42

# trend_pct and 30-day window columns leak the trend_direction label
LEAK = [
    "trend_pct",
    "trend_direction",
    "impressions_last_30d",
    "clicks_last_30d",
    "sessions_last_30d",
    "impressions_prev_30d",
    "clicks_prev_30d",
    "sessions_prev_30d",
]
IDS = ["content_id", "client_id"]
DROP = LEAK + IDS + ["y", "is_declining_label"]


def prepare(df: pd.DataFrame):
    """Return X (numeric features only, fillna(0)), y (1 = declining), groups (client), feature lists.

    Matches the notebook: numeric columns only, NaNs filled with 0, no categorical encodings.
    """
    df = df.copy()
    df["y"] = (df["trend_direction"].str.lower() == "down").astype(int)

    # Exclude drop and leakage columns
    feats = [c for c in df.columns if c not in DROP]

    # Keep ONLY numeric columns (23 signals, exactly as in notebook)
    num_cols = df[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = []  # No categorical columns used in this model

    X = df[num_cols].fillna(0)

    return X, df["y"], df["client_id"], num_cols, cat_cols


def new_model():
    """RandomForestClassifier matching the notebook's exact configuration."""
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )


def _pr(s: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1]."""
    return (s.rank(method="average") - 1) / (len(s) - 1)


def baseline_score(d: pd.DataFrame) -> pd.Series:
    """Week-4 deterministic refresh score (same weights as scripts/02_baseline_score.py)."""
    vis = _pr(np.log1p(d["impressions_90d"]))
    fresh = _pr(d["days_since_last_update"])
    pos = d["avg_position"].clip(1, 50)
    pos_opp = (1 - (pos - pos.min()) / (pos.max() - pos.min())) * vis * (d["avg_position"] > 0)
    depth = (1 - _pr(d["word_count"].fillna(0))) * vis
    return (0.40 * vis + 0.30 * fresh + 0.25 * pos_opp + 0.05 * depth).clip(0, 1)


def precision_at_k(y_true, scores, k: int = 50) -> float:
    """Precision@k – proportion of true positives among the top-k highest scores."""
    idx = np.argsort(np.asarray(scores))[::-1][:k]
    return float(np.asarray(y_true)[idx].mean())


def metrics(y_true, scores, ks=(50,)):
    """Compute ROC-AUC and Precision@50."""
    out = {"ROC-AUC": roc_auc_score(y_true, scores)}
    for k in ks:
        out[f"Precision@{k}"] = precision_at_k(y_true, scores, k)
    return out


def evaluate(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Compare baseline vs Random Forest on the stratified 70/30 split."""
    rows = []
    tr, va = train_test_split(X.index, test_size=0.3, stratify=y, random_state=SEED)

    # Train Random Forest
    rf = new_model().fit(X.loc[tr], y.loc[tr])

    # Validation metrics
    rows.append({
        "model": "Week-4 baseline",
        **metrics(y.loc[va], baseline_score(df.loc[va])),
    })
    rows.append({
        "model": "Random Forest",
        **metrics(y.loc[va], rf.predict_proba(X.loc[va])[:, 1]),
    })

    return pd.DataFrame(rows)


def out_of_fold_risk(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Score all content items using the Random Forest trained on 70% stratified train split."""
    tr_idx, va_idx = train_test_split(X.index, test_size=0.3, stratify=y, random_state=SEED)
    rf = new_model().fit(X.loc[tr_idx], y.loc[tr_idx])
    return pd.Series(rf.predict_proba(X)[:, 1], index=X.index)


def holdout_importance(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_repeats: int = 30):
    """Permutation importance (drop in ROC-AUC) on the 30% stratified validation set."""
    tr_idx, va_idx = train_test_split(X.index, test_size=0.3, stratify=y, random_state=SEED)
    rf = new_model().fit(X.loc[tr_idx], y.loc[tr_idx])
    r = permutation_importance(
        rf,
        X.loc[va_idx],
        y.loc[va_idx],
        n_repeats=n_repeats,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": X.columns,
            "importance": r.importances_mean,
            "std": r.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def default_row(X: pd.DataFrame, num_cols: list, cat_cols: list):
    """Return median baseline profile across all numeric features."""
    return {c: float(X[c].median()) for c in num_cols}


def predict_one(model, X_ref: pd.DataFrame, row: dict) -> float:
    """Predict decline probability for a single dictionary of numeric inputs."""
    r = pd.DataFrame([row])[X_ref.columns]
    return float(model.predict_proba(r)[:, 1][0])