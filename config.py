"""Central configuration for the E-commerce ANN project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ecommerce_sales_34500.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---- Target definitions -------------------------------------------------
# Model 1: profitable (profit_margin > 0)  vs  loss-making (profit_margin <= 0)
PROFIT_THRESHOLD = 0.0
# Model 2: "fast" delivery = delivery_time_days <= FAST_DELIVERY_DAYS, else "slower"
# (median delivery time is 5 days; <=4 gives a balanced 43% / 57% split)
FAST_DELIVERY_DAYS = 4

# Model 3: Return Risk
RETURN_POSITIVE_LABEL = "Returned"
RETURN_NEGATIVE_LABEL = "Not Returned"

# ---- Feature selection (HistGradientBoosting) ---------------------------
SELECTION_TOLERANCE = 0.001   # keep smallest feature set within 0.1 pt of best accuracy
MIN_FEATURES = 3              # never go below this many features

# ---- ANN hyper-parameter search ----------------------------------------
SEARCH_ITER = 8               # random-search candidates
SEARCH_CV = 3
CV_FOLDS = 5                  # final cross-validation folds

# ---- Features hidden from each model -----------------------------------
# Shipping cost is not known when an order is placed and (together with category) almost
# fully determines profit, which made Model 1 ~100% sure of everything. Set a list to []
# to bring these features back (accuracy goes up to ~99.9% for Model 1, probabilities saturate).
SHIPPING_FEATURES = [
    "shipping_cost",
    "shipping_ratio",
    "shipping_per_unit"
]

EXCLUDE_FEATURES = {
    "profit": SHIPPING_FEATURES,
    "delivery": SHIPPING_FEATURES,
    "return": SHIPPING_FEATURES
}
