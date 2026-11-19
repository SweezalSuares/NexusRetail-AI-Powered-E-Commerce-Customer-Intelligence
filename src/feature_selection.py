"""Feature selection driven by HistGradientBoosting.

1. Rank every engineered feature with permutation importance (log-loss based) of a
   HistGradientBoostingClassifier on a validation split of the TRAINING data.
2. Walk down the ranking, fitting a HGB on the top-k features, and record validation
   accuracy / AUC for every k.
3. Keep the smallest k whose accuracy is within SELECTION_TOLERANCE of the best.
The test set is never touched here.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
from src.data_prep import CATEGORICAL_FEATURES


def _hgb():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=150, learning_rate=0.1,
        random_state=config.RANDOM_STATE,
    )


def _as_category(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def select_features( X_train_fe: pd.DataFrame,y_train: pd.Series,selection_metric="accuracy"):
    Xc = _as_category(X_train_fe)
    Xa, Xv, ya, yv = train_test_split(
        Xc, y_train, test_size=0.25, stratify=y_train, random_state=config.RANDOM_STATE)

    base = _hgb().fit(Xa, ya)
    pi = permutation_importance(base, Xv, yv, scoring="neg_log_loss", n_repeats=5,
                                random_state=config.RANDOM_STATE, n_jobs=1)
    ranking = (pd.DataFrame({"feature": Xc.columns, "importance": pi.importances_mean,
                             "std": pi.importances_std})
               .sort_values("importance", ascending=False).reset_index(drop=True))

    curve = []
    for k in range(1, len(ranking) + 1):
        cols = ranking["feature"].iloc[:k].tolist()
        m = _hgb().fit(Xa[cols], ya)
        proba = m.predict_proba(Xv[cols])[:, 1]
        curve.append({"k": k, "feature_added": cols[-1],
                      "accuracy": float(accuracy_score(yv, proba >= 0.5)),
                      "roc_auc": float(roc_auc_score(yv, proba))})
    if selection_metric == "roc_auc":
        best_score = max(c["roc_auc"] for c in curve)

        k_star = next(
            c["k"] for c in curve
            if c["roc_auc"] >= best_score - config.SELECTION_TOLERANCE
        )
    else:
        best_score = max(c["accuracy"] for c in curve)

        k_star = next(
            c["k"] for c in curve
            if c["accuracy"] >= best_score - config.SELECTION_TOLERANCE
        )

    k_star = max(k_star, config.MIN_FEATURES)

    selected = ranking["feature"].iloc[:k_star].tolist()
    selected = ranking["feature"].iloc[:k_star].tolist()
    return selected, ranking, curve
