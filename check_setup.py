"""Run:  python check_setup.py   - verifies libraries, models and predictions on your machine."""
import sys
import sklearn, pandas, numpy

print("python", sys.version.split()[0], "| sklearn", sklearn.__version__,
      "| pandas", pandas.__version__, "| numpy", numpy.__version__)
try:
    import streamlit, plotly
    print("streamlit", streamlit.__version__, "| plotly", plotly.__version__)
except ImportError as e:
    print("MISSING:", e, "-> pip install -r requirements.txt")

from src.predict import predict_orders, version_problem
from src.data_prep import load_data, make_profit_target, make_delivery_target, RAW_INPUT_COLUMNS

print("version check:", version_problem() or "OK")
df = load_data().sample(2000, random_state=1)
res = predict_orders(df[RAW_INPUT_COLUMNS])
pa = (res.profit_ann_prediction.eq("Profitable") == make_profit_target(df).astype(bool)).mean()
da = (res.delivery_ann_prediction.eq("Fast") == make_delivery_target(df).astype(bool)).mean()
print(f"Model 1 accuracy on 2,000 orders: {pa:.3f}   (expected ~0.99)")
print(f"Model 2 accuracy on 2,000 orders: {da:.3f}   (expected ~0.74-0.76)")
