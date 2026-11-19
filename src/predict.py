"""Predict on new orders (library + CLI).

    python -m src.predict --input data/sample_new_orders.csv --output predictions.csv
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config
from src.data_prep import REQUIRED_NEW_DATA_COLUMNS

LABELS = {
    "profit": ("Loss-making", "Profitable"),
    "delivery": ("Slower", "Fast"),
    "return": ("Not Returned", "Returned")
}

def version_problem():
    """Returns a message if the saved models were built with another scikit-learn version."""
    import sklearn
    meta = config.MODELS_DIR / "meta.json"
    if not meta.exists():
        return None
    saved = json.load(open(meta)).get("sklearn_version")
    if saved and saved != sklearn.__version__:
        return (f"Models were trained with scikit-learn {saved} but {sklearn.__version__} is installed. "
                "Run `python run_training.py` to rebuild them for this environment.")
    return None


_CACHE = {}


def load_models(key):
    """Returns (ann_pipeline, hgb_pipeline) for task key 'profit' or 'delivery'."""
    if key not in _CACHE:
        try:
            _CACHE[key] = (joblib.load(config.MODELS_DIR / f"{key}_ann.joblib"),
                           joblib.load(config.MODELS_DIR / f"{key}_hgb.joblib"))
        except Exception as e:
            raise RuntimeError(f"Could not load the {key} models ({e}). "
                               f"{version_problem() or 'Run `python run_training.py` to rebuild them.'}") from e
    return _CACHE[key]


def clear_model_cache():
    """Drop loaded models after a training run replaces the saved artifacts."""
    _CACHE.clear()


def load_metrics(key):
    with open(config.REPORTS_DIR / f"{key}_metrics.json") as f:
        return json.load(f)


def validate_columns(df):
    missing = [c for c in REQUIRED_NEW_DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def predict_orders(
    df: pd.DataFrame,
    threshold: float = None,
    tasks=None
) -> pd.DataFrame:

    validate_columns(df)
    out = df.copy()

    for key, (neg, pos) in LABELS.items():

        if tasks is not None and key not in tasks:
            continue

        ann, hgb = load_models(key)

        p_ann = ann.predict_proba(df)[:, 1]
        p_hgb = hgb.predict_proba(df)[:, 1]

        # Use the saved model-specific threshold unless overridden.
        model_threshold = threshold

        if model_threshold is None:
            model_threshold = load_metrics(key).get(
                "decision_threshold", 0.5
            )

        out[f"{key}_ann_prob_{pos.lower().replace(' ', '_')}"] = np.clip(
            p_ann, 0.05, 0.95
        ).round(4)

        out[f"{key}_ann_prediction"] = [
            pos if p >= model_threshold else neg
            for p in p_ann
        ]

        out[f"{key}_hgb_prob_{pos.lower().replace(' ', '_')}"] = np.clip(
            p_hgb, 0.05, 0.95
        ).round(4)

        out[f"{key}_hgb_prediction"] = [
            pos if p >= 0.5 else neg
            for p in p_hgb
        ]

    return out


def price_sweep(row: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """P(profitable) for every category over a range of unit prices (other inputs = the given order)."""
    import numpy as np
    from src.data_prep import CATEGORY_CHOICES
    prices = np.round(np.geomspace(1, 400, n), 2)
    parts = []
    for c in CATEGORY_CHOICES:
        d = pd.concat([row] * n, ignore_index=True)
        d["category"] = c
        d["price"] = prices
        d["total_amount"] = prices * d["quantity"] * (1 - d["discount"])
        parts.append(d)
    big = pd.concat(parts, ignore_index=True)
    res = predict_orders(big, tasks=["profit"])
    return pd.DataFrame({"category": big["category"], "price": big["price"],
                         "p_profitable": res["profit_ann_prob_profitable"].values})


def sensitivity(key: str, row: pd.DataFrame) -> pd.DataFrame:
    """For one order: how far can each single input move the ANN probability (others held fixed)?"""
    import numpy as np
    from src.data_prep import CATEGORY_CHOICES, GENDER_CHOICES, PAYMENT_CHOICES, REGION_CHOICES
    pos = LABELS[key][1].lower()
    r0 = row.iloc[0]
    year = pd.Timestamp(r0["order_date"]).year
    options = {
        "category": CATEGORY_CHOICES,
        "price": list(np.round(np.geomspace(1, 500, 14), 2)),
        "discount": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        "quantity": [1, 2, 3, 4, 5],
        "payment_method": PAYMENT_CHOICES,
        "region": REGION_CHOICES,
        "customer_age": [18, 25, 35, 45, 55, 65],
        "customer_gender": GENDER_CHOICES,
        "order_date": [pd.Timestamp(year, m, 1) for m in range(1, 13)],
    }
    out = []
    for col, vals in options.items():
        d = pd.concat([row] * len(vals), ignore_index=True)
        d[col] = vals
        d["total_amount"] = d["price"] * d["quantity"] * (1 - d["discount"])
        p = predict_orders(d, tasks=[key])[f"{key}_ann_prob_{pos}"].values
        out.append({"input": col, "p_min": float(p.min()), "p_max": float(p.max()),
                    "swing": float(p.max() - p.min())})
    return pd.DataFrame(out).sort_values("swing", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    a = ap.parse_args()
    res = predict_orders(pd.read_csv(a.input))
    res.to_csv(a.output, index=False)
    print(res.head(10).to_string())
    print(f"\nSaved {len(res)} predictions to {a.output}")
