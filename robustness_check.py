"""
robustness_check.py
===================

How much of the notebook's single-split result survives stricter evaluation?

The notebook (and the original report) score every model on ONE random 80/20
split of 1,010 maps. This script re-scores the same models four ways:

  A. that same split, for reference;
  B. 30 different random 80/20 splits, to see how much one split can vary;
  C. 30 grouped splits, keeping all maps of a series (same matchID) on the
     same side, so a team's other maps from that day are never in training;
  D. a forward-in-time split: HLTV match IDs increase over time, so training
     on the older 80% of series and testing on the newest 20% approximates
     predicting future matches.

It also checks:

  E. whether feature pruning helps when the features are chosen WITHOUT
     looking at the test set (the notebook chooses them on the test set);
  F. a two-feature logistic regression (rating difference and map win-rate
     difference) as a simple baseline for the 142-feature models.

Hyperparameters are the ones the notebook's searches selected, fixed here so
the script runs in a few minutes. Run:  python robustness_check.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import cs2_data as data

warnings.filterwarnings("ignore")

N_REPEATS = 30


# ---------------------------------------------------------------------------
# Models (settings selected by the notebook's cross-validated searches)
# ---------------------------------------------------------------------------

def two_feature_logreg():
    return Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression())])


def l1_logreg():
    return Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(solver="liblinear", penalty="l1",
                                                   C=0.05, class_weight="balanced"))])


def xgboost():
    return XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=4,
                         subsample=0.8, colsample_bytree=1.0, gamma=1,
                         eval_metric="logloss", random_state=0)


def lightgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.01, max_depth=5,
                          num_leaves=60, min_data_in_leaf=20, subsample=0.8,
                          colsample_bytree=0.6, class_weight="balanced",
                          random_state=42, verbose=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_score(make_model, X, y, train_idx, test_idx):
    """Fit on train_idx, return (accuracy, ROC AUC) on test_idx."""
    model = make_model().fit(X.iloc[train_idx], y[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    return accuracy_score(y[test_idx], proba >= 0.5), roc_auc_score(y[test_idx], proba)


def summarise(label, scores):
    s = np.array(scores)
    if len(s) == 1:
        print(f"  {label:24s} accuracy {s[0, 0]:.3f}   AUC {s[0, 1]:.3f}")
        return
    print(f"  {label:24s} accuracy {s[:, 0].mean():.3f} +/- {s[:, 0].std():.3f}   "
          f"AUC {s[:, 1].mean():.3f} +/- {s[:, 1].std():.3f}")


def pruned_lightgbm(X, y, groups, train_idx, test_idx, select_on_test, seed):
    """Prune features with permutation importance (> 0.001 AUC), then refit.

    select_on_test=True  reproduces the notebook: importance measured on the test set.
    select_on_test=False measures it on a held-out slice of the TRAINING data only.
    """
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    if select_on_test:
        model = lightgbm().fit(X_tr, y_tr)
        X_sel, y_sel = X.iloc[test_idx], y[test_idx]
    else:
        inner_tr, inner_val = next(GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed).split(X_tr, y_tr, groups[train_idx]))
        model = lightgbm().fit(X_tr.iloc[inner_tr], y_tr[inner_tr])
        X_sel, y_sel = X_tr.iloc[inner_val], y_tr[inner_val]
    perm = permutation_importance(model, X_sel, y_sel, scoring="roc_auc",
                                  n_repeats=5, random_state=0)
    keep = X.columns[perm.importances_mean > 0.001]
    refit = lightgbm().fit(X_tr[keep], y_tr)
    proba = refit.predict_proba(X.iloc[test_idx][keep])[:, 1]
    return accuracy_score(y[test_idx], proba >= 0.5), roc_auc_score(y[test_idx], proba)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = data.load_features()
    y = df[data.TARGET].to_numpy()
    groups = df["matchID"].to_numpy()
    idx = np.arange(len(df))

    X_all = df.drop(columns=["matchID", "team1", "team2", "map", data.TARGET])
    X_two = pd.DataFrame({
        "rating_diff": df["T1_Rating_2.0_mean"] - df["T2_Rating_2.0_mean"],
        "winrate_diff": df["T1_mapwinrate"] - df["T2_mapwinrate"],
    })
    models = {
        "Two-feature LogReg": (two_feature_logreg, X_two),
        "L1 LogReg (142 feat.)": (l1_logreg, X_all),
        "XGBoost (142 feat.)": (xgboost, X_all),
        "LightGBM (142 feat.)": (lightgbm, X_all),
    }

    print(f"{len(df)} maps from {df['matchID'].nunique()} series; "
          f"team 1 wins {y.mean():.1%} of maps.\n")

    # A. the notebook's split
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    print(f"A. Notebook split (random_state=42, {len(te)} test maps)")
    for name, (make, X) in models.items():
        summarise(name, [fit_score(make, X, y, tr, te)])

    # B. repeated random splits
    print(f"\nB. {N_REPEATS} random 80/20 splits (mean +/- sd)")
    splits = [train_test_split(idx, test_size=0.2, random_state=s, stratify=y)
              for s in range(N_REPEATS)]
    for name, (make, X) in models.items():
        summarise(name, [fit_score(make, X, y, tr, te) for tr, te in splits])

    # C. grouped by series
    print(f"\nC. {N_REPEATS} grouped 80/20 splits, series kept together (mean +/- sd)")
    grouped = list(GroupShuffleSplit(n_splits=N_REPEATS, test_size=0.2,
                                     random_state=0).split(X_all, y, groups))
    for name, (make, X) in models.items():
        summarise(name, [fit_score(make, X, y, tr, te) for tr, te in grouped])

    # D. forward in time
    series = np.sort(df["matchID"].unique())
    cutoff = series[int(len(series) * 0.8)]
    tr, te = idx[df["matchID"] < cutoff], idx[df["matchID"] >= cutoff]
    print(f"\nD. Train on older series, test on the newest 20% ({len(te)} test maps)")
    for name, (make, X) in models.items():
        summarise(name, [fit_score(make, X, y, tr, te)])

    # E. pruning with and without test-set selection
    X_lgb = X_all.copy()
    X_lgb["map"] = df["map"].astype("category")
    print(f"\nE. Pruned LightGBM over the {N_REPEATS} grouped splits of C")
    summarise("features chosen on test",
              [pruned_lightgbm(X_lgb, y, groups, tr, te, True, s)
               for s, (tr, te) in enumerate(grouped)])
    summarise("features chosen on train",
              [pruned_lightgbm(X_lgb, y, groups, tr, te, False, s)
               for s, (tr, te) in enumerate(grouped)])


if __name__ == "__main__":
    main()
