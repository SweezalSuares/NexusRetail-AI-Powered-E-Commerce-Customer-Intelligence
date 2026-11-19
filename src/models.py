"""Pipelines: feature engineering -> (selected) preprocessing -> ANN or HistGradientBoosting."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

import config
from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features


def _split(selected):
    num = [f for f in NUMERIC_FEATURES if f in selected]
    cat = [f for f in CATEGORICAL_FEATURES if f in selected]
    return num, cat


def build_ann(selected, **mlp_params) -> Pipeline:
    num, cat = _split(selected)
    blocks = []
    if num:
        blocks.append(("num", StandardScaler(), num))
    if cat:
        blocks.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    params = dict(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                  alpha=1e-3, learning_rate_init=1e-3, batch_size=256, max_iter=300,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                  random_state=config.RANDOM_STATE)
    params.update(mlp_params)
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features)),
        ("prep", ColumnTransformer(blocks)),
        ("ann", MLPClassifier(**params)),
    ])


def build_hgb(selected) -> Pipeline:
    num, cat = _split(selected)
    blocks = []
    if num:
        blocks.append(("num", "passthrough", num))
    if cat:
        blocks.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=np.nan), cat))
    mask = [False] * len(num) + [True] * len(cat)
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features)),
        ("prep", ColumnTransformer(blocks)),
        ("hgb", HistGradientBoostingClassifier(
            categorical_features=mask, max_iter=250, learning_rate=0.08,
            random_state=config.RANDOM_STATE)),
    ])
