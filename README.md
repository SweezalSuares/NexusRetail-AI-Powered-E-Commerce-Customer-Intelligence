# NexusRetail: Customer Intelligence with ANN / Multilayer Perceptrons

Three artificial neural networks (ANN / multi-layer perceptron) trained on `ecommerce_sales_34500.csv`
(34,500 orders), with **feature selection driven by HistGradientBoosting**, plus a **Streamlit dashboard**.

| Model | Prediction | Type | Target definition |
|---|---|---|---|
| 1 | Order profitable vs loss-making | Binary ANN classifier | `profit_margin > 0` |
| 2 | Order delivered fast vs slower | Binary ANN classifier | `delivery_time_days <= 4` (change in `config.py`) |
| 3 | Order returned vs not returned | Binary ANN classifier | `returned` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_training.py            # trains all models (a few minutes). Use --quick for a smoke test
python run_training.py --seed 123 # optionally set the random seed for splits and model training
streamlit run app/streamlit_app.py
```

Trained models are already included in `models/`. If your scikit-learn version differs and loading fails,
run `python run_training.py` once to rebuild them. The dashboard sidebar can also retrain all models using
an adjustable random seed.

Score new orders from the command line:

```bash
python -m src.predict --input data/sample_new_orders.csv --output predictions.csv
```

## Project layout

```
config.py                 paths, target definitions, selection / search settings
run_training.py           entry point for training
data/                     dataset + sample_new_orders.csv
src/data_prep.py          loading, targets, feature engineering (17 numeric + 4 categorical features)
src/feature_selection.py  HistGradientBoosting permutation-importance ranking + feature-count curve
src/models.py             ANN (MLP) and HistGradientBoosting pipelines
src/train.py              selection -> ANN random search -> CV -> evaluation -> save
src/evaluate.py           accuracy / precision / recall / F1 / AUC / MCC / curves
src/plots.py              static PNG figures
src/predict.py            predict new data (library + CLI)
app/streamlit_app.py      dashboard
models/                   saved pipelines (*.joblib)
reports/                  *_metrics.json, *_test_predictions.csv, figures/*.png
```

## Method

1. **No leakage** - only order-time columns are inputs. `returned`, `delivery_time_days` and
   `profit_margin` are outcomes and are excluded. IDs and dates-as-strings are dropped; dates become
   month / weekday / year / quarter / days-since-start.
2. **Feature engineering** - log price, gross amount, discount amount, shipping ratio, shipping per unit, etc.
3. **Feature selection** - a HistGradientBoostingClassifier is fitted on a validation split of the *training*
   data; features are ranked by permutation importance, and the smallest top-k set within 0.1 pt of the
   best validation accuracy is kept (never fewer than 3). The test set is never used.
4. **ANN** - StandardScaler + OneHotEncoder -> MLP (ReLU, Adam, L2 regularisation, early stopping), with a
   randomised hyper-parameter search (hidden layers, alpha, learning rate, batch size).
5. **Evaluation** - 80/20 stratified split, 5-fold CV, and comparison against HistGradientBoosting and
   against the ANN without feature selection.

## Dashboard

* **Overview** - headline results and dataset explorer charts
* **Model 1 / Model 2** - accuracy, precision, recall, F1, ROC-AUC, MCC; confusion matrix; ROC and
  precision-recall curves; probability histograms; interactive **threshold slider**; feature importance and
  feature-count curve; model comparison; CV scores
* **Predict · Profit / Loss** and **Predict · Delivery speed** - two separate pages, each with its own entry form
  (gauge, verdict, what-if chart) and its own batch CSV upload with downloadable predictions

## Check your setup

```bash
python check_setup.py     # prints library versions and tests both models on 2,000 real orders
```

## Note on reachable accuracy

Shipping features (`shipping_cost`, ratio, per-unit) are hidden from both models (`config.EXCLUDE_FEATURES`)
because they are not known when an order is placed. With them, Model 1 reaches ~99.9% but is 100% sure of
almost every order, because shipping cost relative to order value nearly determines profit. Without them
Model 1 reaches ~95.5%, driven by category and price. Getting it down to 80-90% would mean discarding category
(the strongest legitimate signal), which makes the model worse rather than more realistic.
In this dataset, delivery time depends almost entirely on **region** (Central/North ~4 days, South/West ~5,
East ~6) plus random variation; no other column carries signal. That caps Model 2 at roughly 75-76% accuracy for
the "<= 4 days" definition, for both the ANN and gradient boosting. Re-defining the classes can raise the number
(e.g. "<= 3 days" gives ~86%) but only because the classes become imbalanced and "always slower" already scores
that much, so it is not a real improvement.
