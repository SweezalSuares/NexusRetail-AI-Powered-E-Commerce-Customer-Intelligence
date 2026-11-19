"""Data loading, target creation and feature engineering."""
import numpy as np
import pandas as pd

import config
RAW_INPUT_COLUMNS = [
    "category", "price", "discount", "quantity", "payment_method", "order_date",
    "region", "total_amount", "shipping_cost", "customer_age", "customer_gender",
]
# total_amount is computed if absent; shipping_cost is optional (not used by default, see config.EXCLUDE_FEATURES)
REQUIRED_NEW_DATA_COLUMNS = [c for c in RAW_INPUT_COLUMNS if c not in ("total_amount", "shipping_cost")]

CATEGORICAL_FEATURES = ["category", "payment_method", "region", "customer_gender"]
NUMERIC_FEATURES = [
    "price", "log_price", "discount", "quantity", "gross_amount", "discount_amount",
    "total_amount", "shipping_cost", "shipping_ratio", "shipping_per_unit",
    "customer_age", "order_month", "order_dow", "order_year", "is_weekend",
    "order_quarter", "days_since_ref",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REF_DATE = pd.Timestamp("2023-01-01")

CATEGORY_CHOICES = ["Beauty", "Electronics", "Fashion", "Grocery", "Home", "Sports", "Toys"]
PAYMENT_CHOICES = ["COD", "Credit Card", "Debit Card", "PayPal", "UPI", "Wallet"]
REGION_CHOICES = ["Central", "East", "North", "South", "West"]
GENDER_CHOICES = ["Female", "Male", "Other"]


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    return df


def make_profit_target(df: pd.DataFrame) -> pd.Series:
    """1 = profitable, 0 = loss-making."""
    return (df["profit_margin"] > config.PROFIT_THRESHOLD).astype(int)


def make_delivery_target(df: pd.DataFrame) -> pd.Series:
    """1 = fast delivery, 0 = slower delivery."""
    return (df["delivery_time_days"] <= config.FAST_DELIVERY_DAYS).astype(int)

def make_return_target(df: pd.DataFrame) -> pd.Series:
    """
    1 = returned
    0 = not returned
    """
    if "returned" not in df.columns:
        raise ValueError("Dataset must contain a 'returned' column.")

    values = df["returned"]

    if values.isna().any():
        raise ValueError("The 'returned' column contains missing values.")

    normalized = values.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "returned": 1,
        "not returned": 0,
    }

    target = normalized.map(mapping)

    if target.isna().any():
        unknown = normalized[target.isna()].unique().tolist()
        raise ValueError(f"Unknown values in returned column: {unknown}")

    return target.astype(int)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw order columns -> model features. Used as the first step of every pipeline."""
    d = df.copy()
    d["order_date"] = pd.to_datetime(d["order_date"])
    if "total_amount" not in d.columns:
        d["total_amount"] = d["price"] * d["quantity"] * (1 - d["discount"])

    out = pd.DataFrame(index=d.index)
    out["price"] = d["price"].astype(float)
    out["log_price"] = np.log1p(out["price"])
    out["discount"] = d["discount"].astype(float)
    out["quantity"] = d["quantity"].astype(float)
    out["gross_amount"] = out["price"] * out["quantity"]
    out["discount_amount"] = out["gross_amount"] * out["discount"]
    out["total_amount"] = d["total_amount"].astype(float)
    out["shipping_cost"] = d["shipping_cost"].astype(float) if "shipping_cost" in d.columns else np.nan
    out["shipping_ratio"] = out["shipping_cost"] / out["total_amount"].clip(lower=0.01)
    out["shipping_per_unit"] = out["shipping_cost"] / out["quantity"].clip(lower=1)
    out["customer_age"] = d["customer_age"].astype(float)
    out["order_month"] = d["order_date"].dt.month.astype(float)
    out["order_dow"] = d["order_date"].dt.dayofweek.astype(float)
    out["order_year"] = d["order_date"].dt.year.astype(float)
    out["is_weekend"] = (d["order_date"].dt.dayofweek >= 5).astype(float)
    out["order_quarter"] = d["order_date"].dt.quarter.astype(float)
    out["days_since_ref"] = (d["order_date"] - REF_DATE).dt.days.astype(float)
    for c in CATEGORICAL_FEATURES:
        out[c] = d[c].astype(str)
    return out[ALL_FEATURES]
