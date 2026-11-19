"""Train both ANN models end-to-end:  feature selection (HGB) -> ANN search -> evaluation."""
import argparse
import json
import time

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split
)
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

import config
from src.data_prep import (
    RAW_INPUT_COLUMNS,
    engineer_features,
    load_data,
    make_delivery_target,
    make_profit_target,
    make_return_target
)
from src.evaluate import full_metrics
from src.feature_selection import select_features
from src.models import build_ann, build_hgb
from src.plots import make_figures

TASKS = {
    "profit": {
        "title": "Model 1 - Profitable vs Loss-making",
        "target_fn": make_profit_target,
        "positive_label": "Profitable", "negative_label": "Loss-making",
        "definition": f"Profitable = profit_margin > {config.PROFIT_THRESHOLD}",
    },
    "delivery": {
        "title": "Model 2 - Fast vs Slower delivery",
        "target_fn": make_delivery_target,
        "positive_label": "Fast", "negative_label": "Slower",
        "definition": f"Fast = delivery_time_days <= {config.FAST_DELIVERY_DAYS}",
    },
    "return": {
    "title": "Model 3 - Return Risk Prediction",
    "target_fn": make_return_target,
    "positive_label": "Returned",
    "negative_label": "Not Returned",
    "definition": "Returned = order was returned"
},
}

PARAM_GRID = {
    "ann__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
    "ann__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "ann__learning_rate_init": [1e-3, 3e-3],
    "ann__batch_size": [128, 256],
}


def _to_py(o):
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def run_task(key, df, quick=False):
    spec = TASKS[key]
    t0 = time.time()
    y = spec["target_fn"](df)
    X = df[RAW_INPUT_COLUMNS]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                                          random_state=config.RANDOM_STATE)
    labels = [spec["negative_label"], spec["positive_label"]]
    print(f"\n=== {spec['title']} ===  train={len(Xtr):,} test={len(Xte):,} "
          f"positive rate={y.mean():.3f}")

    # 1) feature selection with HistGradientBoosting (training data only)
    excluded = config.EXCLUDE_FEATURES.get(key, [])
    X_fe_train = engineer_features(Xtr).drop(columns=excluded)
    selected, ranking, curve = select_features(
    X_fe_train,
    ytr,
    selection_metric="roc_auc" if key == "return" else "accuracy"
)
    print(f"Selected {len(selected)} features: {selected}")

    # 2) ANN hyper-parameter search on selected features
    ann = build_ann(selected)
    scoring = "average_precision" if key == "return" else "accuracy"
    fit_params = {}
    sample_weight = None
    if key == "return":
        sample_weight = compute_sample_weight("balanced", ytr)
        fit_params["ann__sample_weight"] = sample_weight

    search = RandomizedSearchCV(
        ann,
        PARAM_GRID,
        n_iter=2 if quick else config.SEARCH_ITER,
        cv=2 if quick else config.SEARCH_CV,
        scoring=scoring,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )
    search.fit(Xtr, ytr, **fit_params)
    best_params = {k.replace("ann__", ""): v for k, v in search.best_params_.items()}
    ann = search.best_estimator_
    threshold = 0.5

    if key == "return":
        cv_predictions = cross_val_predict(
            clone(ann),
            Xtr,
            ytr,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=config.RANDOM_STATE
            ),
            method="predict_proba",
            params={"ann__sample_weight": sample_weight} if sample_weight is not None else None,
            n_jobs=-1
        )[:, 1]

        precision, recall, thresholds = precision_recall_curve(
            ytr,
            cv_predictions
        )

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-10)
        )

        best_index = np.argmax(f1_scores)
        threshold = float(thresholds[best_index])

        print(f"Selected return-risk threshold: {threshold:.4f}")
    print("Best ANN params:", best_params)

    # 3) cross-validation of the final ANN
    cv = cross_val_score(clone(ann), Xtr, ytr, cv=StratifiedKFold(
        2 if quick else config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="accuracy", params=fit_params or None, n_jobs=-1)

    # 4) comparison models
    all_features = list(X_fe_train.columns)
    ann_all = build_ann(all_features, **best_params)
    hgb_all = build_hgb(all_features)
    hgb_sel = build_hgb(selected)
    if sample_weight is not None:
        ann_all.fit(Xtr, ytr, ann__sample_weight=sample_weight)
        hgb_all.fit(Xtr, ytr, hgb__sample_weight=sample_weight)
        hgb_sel.fit(Xtr, ytr, hgb__sample_weight=sample_weight)
    else:
        ann_all.fit(Xtr, ytr)
        hgb_all.fit(Xtr, ytr)
        hgb_sel.fit(Xtr, ytr)

    probas = {
        "ANN (selected features)": ann.predict_proba(Xte)[:, 1],
        "ANN (all features)": ann_all.predict_proba(Xte)[:, 1],
        "HistGradientBoosting (selected features)": hgb_sel.predict_proba(Xte)[:, 1],
        "HistGradientBoosting (all features)": hgb_all.predict_proba(Xte)[:, 1],
    }
    models_metrics = {
    n: full_metrics(
        yte,
        p,
        labels,
        thr=threshold if n == "ANN (selected features)" else 0.5
    )
    for n, p in probas.items()
    }
    mlp = ann.named_steps["ann"]
    majority = float(max(yte.mean(), 1 - yte.mean()))

    metrics = {
        "task_key": key, "title": spec["title"], "definition": spec["definition"],
        "positive_label": spec["positive_label"], "negative_label": spec["negative_label"],
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "random_seed": config.RANDOM_STATE,
        "class_balance": {spec["positive_label"]: int(y.sum()), spec["negative_label"]: int((1 - y).sum())},
        "majority_baseline_accuracy": majority,
        "candidate_features": all_features, "excluded_features": excluded, "selected_features": selected,
        "feature_ranking": ranking.to_dict("records"), "selection_curve": curve,
        "best_params": best_params, "hidden_layers": list(mlp.hidden_layer_sizes),
        "n_epochs_run": int(mlp.n_iter_),
        "cv": {"scores": cv.tolist(), "mean": float(cv.mean()), "std": float(cv.std())},
        "models": models_metrics, "decision_threshold": threshold,
        "training_seconds": round(time.time() - t0, 1),
    }

    # 5) persist
    config.MODELS_DIR.mkdir(exist_ok=True); config.REPORTS_DIR.mkdir(exist_ok=True)
    with open(config.MODELS_DIR / "meta.json", "w") as f:
        json.dump({"sklearn_version": sklearn.__version__, "random_seed": config.RANDOM_STATE}, f)
    joblib.dump(ann, config.MODELS_DIR / f"{key}_ann.joblib")
    joblib.dump(hgb_sel, config.MODELS_DIR / f"{key}_hgb.joblib")
    pd.DataFrame({"y_true": yte.values, **{n: p for n, p in probas.items()}}).to_csv(
        config.REPORTS_DIR / f"{key}_test_predictions.csv", index=False)
    with open(config.REPORTS_DIR / f"{key}_metrics.json", "w") as f:
        json.dump(_to_py(metrics), f, indent=1)
    make_figures(key, _to_py(metrics), yte.values, probas["ANN (selected features)"],
                 mlp.loss_curve_, mlp.validation_scores_)

    p = models_metrics["ANN (selected features)"]
    print(f"ANN test  acc={p['accuracy']:.4f} prec={p['precision']:.4f} rec={p['recall']:.4f} "
          f"f1={p['f1']:.4f} auc={p['roc_auc']:.4f} | CV {cv.mean():.4f}+/-{cv.std():.4f} "
          f"| majority baseline {majority:.4f}")
    for n, r in models_metrics.items():
        print(f"   {n:45s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny search, for smoke tests")
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE, help="random seed for data splits and models")
    a = ap.parse_args()
    if a.seed < 0:
        ap.error("--seed must be non-negative")
    config.RANDOM_STATE = a.seed
    df = load_data()
    for k in (TASKS if a.task == "all" else [a.task]):
        run_task(k, df, quick=a.quick)
    print("\nDone. Models in /models, metrics in /reports.")


if __name__ == "__main__":
    main()
