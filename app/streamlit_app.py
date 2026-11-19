"""Streamlit dashboard: model metrics, graphs, and separate prediction pages for each model.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import datetime as dt
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.data_prep import (CATEGORY_CHOICES, GENDER_CHOICES, PAYMENT_CHOICES, REGION_CHOICES,
                           REQUIRED_NEW_DATA_COLUMNS, load_data)
from src.predict import (LABELS, clear_model_cache, load_metrics, predict_orders, price_sweep,
                         sensitivity, version_problem)

st.set_page_config(page_title="NexusRetail: Customer Intelligence", layout="wide")

# ------------------------------------------------------------------ theme
PALETTE = ["#4F46E5", "#06B6D4", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899"]
INDIGO, GREEN, RED = "#4F46E5", "#10B981", "#EF4444"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #06B6D4 100%);
       padding: 1.6rem 1.9rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
       box-shadow: 0 10px 30px rgba(79,70,229,.25);}
.hero h1 {color: white; margin: 0; font-size: 2rem; padding: 0;}
.hero p {color: rgba(255,255,255,.88); margin: .35rem 0 0 0; font-size: 1rem;}
[data-testid="stMetric"] {background: white; border-radius: 14px; padding: 14px 16px;
       box-shadow: 0 2px 12px rgba(31,41,55,.07); border-left: 5px solid #4F46E5;}
[data-testid="stMetricLabel"] {color: #6B7280;}
[data-testid="stMetricValue"] {color: #1F2937; font-weight: 700;}
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E5E7EB;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #EEF0F8; border-radius: 10px 10px 0 0; padding: 8px 18px;}
.stTabs [aria-selected="true"] {background: #4F46E5 !important; color: white !important;}
.stTabs [aria-selected="true"] p {color: white !important;}
.verdict {border-radius: 16px; padding: 1.1rem 1.4rem; font-size: 1.35rem; font-weight: 700;
          margin: .4rem 0 .6rem 0; text-align: center;}
.verdict.good {background: #D1FAE5; color: #065F46; border: 2px solid #10B981;}
.verdict.bad {background: #FEE2E2; color: #991B1B; border: 2px solid #EF4444;}
.card {background: white; border-radius: 14px; padding: 1rem 1.2rem;
       box-shadow: 0 2px 12px rgba(31,41,55,.07);}
</style>
""", unsafe_allow_html=True)

TASK_TITLES = {
    "profit": "Model 1 · Profit vs Loss",
    "delivery": "Model 2 · Fast vs Slower delivery",
    "return": "Model 3 · Return Risk"
}
PRIMARY = "ANN (selected features)"

# every plotly chart gets a unique, run-stable key (avoids StreamlitDuplicateElementId)
_key_counter = itertools.count()


def chart(fig, height=None):
    fig.update_layout(font=dict(family="Inter, Segoe UI, sans-serif", size=13),
                      margin=dict(l=20, r=20, t=55, b=30), paper_bgcolor="rgba(0,0,0,0)")
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, key=f"chart_{next(_key_counter)}")


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def artifacts_ready():
    return all((config.MODELS_DIR / f"{k}_ann.joblib").exists() and
               (config.REPORTS_DIR / f"{k}_metrics.json").exists() for k in TASK_TITLES)


@st.cache_data(show_spinner=False)
def get_metrics(key):
    return load_metrics(key)


@st.cache_data(show_spinner=False)
def get_test_predictions(key):
    return pd.read_csv(config.REPORTS_DIR / f"{key}_test_predictions.csv")


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


def saved_random_seed():
    meta_path = config.MODELS_DIR / "meta.json"
    if not meta_path.exists():
        return config.RANDOM_STATE
    with open(meta_path) as f:
        return int(json.load(f).get("random_seed", config.RANDOM_STATE))


def pct(x):
    return f"{x * 100:.2f}%"


def metric_row(m):
    cols = st.columns(6)
    for c, (label, k) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                    ("Recall", "recall"), ("F1-score", "f1"),
                                    ("ROC-AUC", "roc_auc"), ("MCC", "mcc")]):
        c.metric(label, pct(m[k]) if k != "mcc" else f"{m[k]:.3f}")


def confusion_fig(cm, labels, title="Confusion matrix (test set)"):
    cm = np.array(cm)
    txt = [[f"{v:,}" for v in r] for r in cm]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, text=txt, texttemplate="%{text}",
                               textfont=dict(size=18), colorscale=[[0, "#EEF2FF"], [1, INDIGO]],
                               showscale=False))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
    return fig


def confusion_from_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return [[tn, fp], [fn, tp]], acc, prec, rec, f1


# ------------------------------------------------------------------ pages
def page_overview():
    hero("NexusRetail",
         "Customer Intelligence Dashboard · 34,500 orders · ANN and HistGradientBoosting models")
    st.subheader("Executive overview")
    for key in TASK_TITLES:
        m = get_metrics(key); p = m["models"][PRIMARY]
        st.subheader(TASK_TITLES[key])
        st.caption(m["definition"])
        metric_row(p)
        st.caption(
            f"CV accuracy: {pct(m['cv']['mean'])} ± {m['cv']['std'] * 100:.2f} pt · "
            f"Majority baseline: {pct(m['majority_baseline_accuracy'])} · "
            f"Selected features ({len(m['selected_features'])}): "
            + ", ".join(m["selected_features"])
        )
        st.divider()

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Records", "34,500")
    with c2:
        st.metric("Prediction targets", "3")
    with c3:
        st.metric("Train / test split", "80% / 20%")


def page_dataset():
    hero("Dataset",
         "Explore the source data, distributions, and quality checks for NexusRetail")
    df = get_data().copy()
    df["profitable"] = np.where(df["profit_margin"] > config.PROFIT_THRESHOLD, "Profitable", "Loss-making")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Date range", f"{df['order_date'].min().date()} to {df['order_date'].max().date()}")

    st.subheader("About the dataset")
    st.write(
        "The synthetic retail dataset contains order-level commercial, customer, and delivery attributes "
        "for 34,500 ecommerce transactions. It supports three binary prediction tasks while excluding outcome "
        "fields such as delivery time and profit margin from model inputs."
    )

    tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Data sample"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            chart(px.histogram(df, x="total_amount", nbins=50, title="Order value distribution",
                               labels={"total_amount": "Order total"}), 380)
        with c2:
            chart(px.histogram(df, x="delivery_time_days", color="region", barmode="group",
                               title="Delivery time by region"), 380)
        c3, c4 = st.columns(2)
        with c3:
            chart(px.bar(df["category"].value_counts().rename_axis("Category").reset_index(name="Orders"),
                         x="Category", y="Orders", title="Orders by category"), 380)
        with c4:
            chart(px.pie(df, names="payment_method", hole=0.55, title="Payment method mix"), 380)
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            chart(px.box(df, x="region", y="delivery_time_days", color="region",
                         title="Delivery time by region").update_layout(showlegend=False), 380)
        with c2:
            g = df.groupby("category")["profitable"].apply(
                lambda s: (s == "Profitable").mean()
            ).reset_index(name="share")
            chart(px.bar(g, x="category", y="share", title="Profitable order share",
                         labels={"share": "Profitable share", "category": "Category"},
                         range_y=[0, 1]), 380)
        monthly = df.set_index("order_date").resample("MS")["total_amount"].sum().reset_index()
        chart(px.area(monthly, x="order_date", y="total_amount", title="Monthly revenue"), 380)
    with tab3:
        st.dataframe(df.head(200), use_container_width=True)
        duplicates = df[df.duplicated(keep=False)]
        st.subheader("Duplicate records")
        st.metric("Duplicate rows", f"{len(duplicates):,}")
        if duplicates.empty:
            st.info("No duplicate records found in the source dataset.")
        else:
            st.dataframe(duplicates, use_container_width=True)
        st.download_button("Download dataset", df.to_csv(index=False), "nexusretail_dataset.csv", "text/csv")


def page_data_model():
    hero("Data Model",
         "Features, targets, preprocessing, feature selection, and model architecture")
    df = get_data()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Target 1: Profitability")
        st.info("Positive: profit_margin > 0\n\nNegative: profit_margin <= 0")
        pm = get_metrics("profit")
        st.write(f"Selected features: **{len(pm['selected_features'])}**")
        st.write("Top features: " + ", ".join(pm["selected_features"][:5]))

    with c2:
        st.markdown("### Target 2: Delivery speed")
        st.info("Fast: delivery_time_days <= 4\n\nSlower: delivery_time_days > 4")
        dm = get_metrics("delivery")
        st.write(f"Selected features: **{len(dm['selected_features'])}**")
        st.write("Top features: " + ", ".join(dm["selected_features"][:5]))

    with c3:
        st.markdown("### Target 3: Return Risk")
        st.info("Positive: Order is returned\n\nNegative: Order is not returned")
        rm = get_metrics("return")
        st.write(f"Selected features: **{len(rm['selected_features'])}**")
        st.write("Top features: " + ", ".join(rm["selected_features"][:5]))

    st.subheader("Model pipeline")
    pipeline = pd.DataFrame([
        ("1", "Input", "34,500 order-level records"),
        ("2", "Preprocessing", "Numeric scaling and categorical one-hot encoding"),
        ("3", "Feature selection", "HistGradientBoosting permutation importance"),
        ("4", "ANN training", "MLP classifier with early stopping and hyperparameter search"),
        ("5", "Evaluation", "80/20 split, 5-fold cross-validation, test-set metrics"),
        ("6", "Prediction", "Profitability and delivery-speed probability outputs, plus return-risk probability for new orders "),
    ], columns=["Step", "Stage", "Description"])
    st.dataframe(pipeline, hide_index=True, use_container_width=True)

    st.subheader("Data dictionary")
    data_dictionary = pd.DataFrame([
        ("category", "Categorical", "Product category"),
        ("price", "Numeric", "Unit price"),
        ("discount", "Numeric", "Discount rate"),
        ("quantity", "Numeric", "Units ordered"),
        ("payment_method", "Categorical", "Customer payment method"),
        ("order_date", "Date", "Date the order was placed"),
        ("region", "Categorical", "Customer/fulfilment region"),
        ("total_amount", "Numeric", "Net order value"),
        ("customer_age", "Numeric", "Customer age"),
        ("customer_gender", "Categorical", "Customer gender"),
        ("profit_margin", "Numeric", "Target outcome for profitability model"),
        ("delivery_time_days", "Numeric", "Target outcome for delivery model"),
    ], columns=["Field", "Type", "Description"])
    st.dataframe(data_dictionary, hide_index=True, use_container_width=True)
    st.caption("Shipping cost and shipping-derived fields are excluded because they are not known at prediction time.")


def page_model(key):
    m = get_metrics(key)
    labels = [m["negative_label"], m["positive_label"]]
    prim = m["models"][PRIMARY]
    hero(TASK_TITLES[key], f"{m['definition']} · train {m['n_train']:,} / test {m['n_test']:,} orders · "
                           f"ANN layers {m['hidden_layers']} · {m['n_epochs_run']} epochs")
    metric_row(prim)
    st.info(f"5-fold CV accuracy **{pct(m['cv']['mean'])} ± {m['cv']['std'] * 100:.2f}** · "
            f"always-predict-majority baseline **{pct(m['majority_baseline_accuracy'])}**")

    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Threshold explorer",
                                      "Feature selection", "Model comparison"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            chart(confusion_fig(prim["confusion_matrix"], labels), 400)
        with c2:
            fig = go.Figure()
            for name, r in m["models"].items():
                fig.add_scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                                name=f"{name} (AUC {r['roc_auc']:.3f})")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="chance")
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                              yaxis_title="True positive rate", legend=dict(font=dict(size=9), y=0.05, x=0.35))
            chart(fig, 400)
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            for name, r in m["models"].items():
                fig.add_scatter(x=r["pr"]["recall"], y=r["pr"]["precision"], mode="lines", name=name)
            fig.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision",
                              legend=dict(font=dict(size=9)))
            chart(fig, 400)
        with c4:
            tp = get_test_predictions(key).copy()
            tp[PRIMARY] = tp[PRIMARY].clip(0.05, 0.95)
            tp["Actual"] = np.where(tp["y_true"] == 1, m["positive_label"], m["negative_label"])
            chart(px.histogram(tp, x=PRIMARY, color="Actual", nbins=40, barmode="overlay", opacity=0.7,
                               title=f"ANN P({m['positive_label']}) by actual class",
                               labels={PRIMARY: f"P({m['positive_label']})"}), 400)
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(prim["classification_report"]).T.round(4))
        st.subheader("Cross-validation accuracy per fold")
        sc = m["cv"]["scores"]
        chart(px.bar(x=[f"Fold {i + 1}" for i in range(len(sc))], y=sc, text_auto=".3f",
                     labels={"x": "", "y": "Accuracy"}, range_y=[min(sc) - 0.02, min(1.0, max(sc) + 0.01)]), 320)

    with tab2:
        st.write("Move the decision threshold and watch precision and recall trade off on the test set.")
        default_threshold = round(float(m.get("decision_threshold", 0.5)), 2)
        thr = st.slider(f"Classify as **{m['positive_label']}** when probability ≥", 0.01, 0.95,
                        min(max(default_threshold, 0.01), 0.95), 0.01,
                        key=f"thr_{key}")
        tp = get_test_predictions(key)
        cm, acc, prec, rec, f1 = confusion_from_threshold(tp["y_true"].values, tp[PRIMARY].values, thr)
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", pct(acc)); b.metric("Precision", pct(prec))
        c.metric("Recall", pct(rec)); d.metric("F1", pct(f1))
        c1, c2 = st.columns(2)
        with c1:
            chart(confusion_fig(cm, labels, f"Confusion matrix @ threshold {thr:.2f}"), 400)
        with c2:
            sw = pd.DataFrame(prim["threshold_sweep"]).melt("threshold", var_name="metric", value_name="value")
            fig = px.line(sw, x="threshold", y="value", color="metric", title="Metrics vs threshold")
            fig.add_vline(x=thr, line_dash="dash", line_color="gray")
            chart(fig, 400)

    with tab3:
        st.write("Features are ranked by **HistGradientBoosting permutation importance** on a validation split "
                 f"of the training data; the smallest set within {config.SELECTION_TOLERANCE * 100:.1f} pt of the "
                 f"best accuracy is kept (minimum {config.MIN_FEATURES}).")
        if m.get("excluded_features"):
            st.caption("Hidden from this model: " + ", ".join(m["excluded_features"]) +
                       " (not known when an order is placed).")
        c1, c2 = st.columns(2)
        with c1:
            rk = pd.DataFrame(m["feature_ranking"]).head(15)
            rk["selected"] = rk["feature"].isin(m["selected_features"])
            chart(px.bar(rk.iloc[::-1], x="importance", y="feature", orientation="h", color="selected",
                         color_discrete_map={True: GREEN, False: "#CBD5E1"},
                         title="Permutation importance (top 15)"), 460)
        with c2:
            scv = pd.DataFrame(m["selection_curve"])
            fig = px.line(scv, x="k", y=["accuracy", "roc_auc"], markers=True,
                          title="Validation score vs number of top-ranked features")
            fig.add_vline(x=len(m["selected_features"]), line_dash="dash", line_color=RED)
            chart(fig, 460)
        st.success("Selected features: " + ", ".join(m["selected_features"]))

    with tab4:
        rows = [{"Model": n, "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
                 "F1": r["f1"], "ROC-AUC": r["roc_auc"]} for n, r in m["models"].items()]
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df.style.format({c: "{:.4f}" for c in cmp_df.columns[1:]}))
        long = cmp_df.melt("Model", var_name="Metric", value_name="Score")
        chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                     title="Model comparison (test set)", range_y=[0, 1.02]))


# ------------------------------------------------------------------ prediction pages (one per model)
def prob_gauge(p, pos_label, risk=False):
    lower_color, upper_color = (GREEN, RED) if risk else (RED, GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": f"P({pos_label})"},
        gauge={"axis": {"range": [0, 95]}, "bar": {"color": RED if risk else INDIGO},
               "steps": [{"range": [0, 50], "color": lower_color},
                         {"range": [50, 100], "color": upper_color}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
    fig.update_layout(margin=dict(t=60, b=10))
    return fig


def order_form(key):
    """Entry form for this model only. Returns a one-row DataFrame or None."""
    with st.form(f"order_form_{key}"):
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Category", CATEGORY_CHOICES, index=CATEGORY_CHOICES.index("Toys"))
        price = c1.number_input("Unit price", 0.5, 5000.0, 12.0, 0.5)
        quantity = c1.number_input("Quantity", 1, 20, 1)
        discount = c2.select_slider("Discount", [0.0, 0.05, 0.10, 0.15, 0.20, 0.30], value=0.0)
        payment = c2.selectbox("Payment method", PAYMENT_CHOICES, index=1)
        region = c2.selectbox("Region", REGION_CHOICES, index=2)
        age = c3.number_input("Customer age", 16, 100, 35)
        gender = c3.selectbox("Customer gender", GENDER_CHOICES)
        odate = c3.date_input("Order date", dt.date(2025, 6, 15))
        submitted = st.form_submit_button(f"Predict {TASK_TITLES[key].split('·')[1].strip()}", type="primary")
    if not submitted:
        return None
    total = round(price * quantity * (1 - discount), 2)
    return pd.DataFrame([{"category": category, "price": price, "discount": discount, "quantity": quantity,
                          "payment_method": payment, "order_date": pd.Timestamp(odate), "region": region,
                          "total_amount": total, "customer_age": age, "customer_gender": gender}])


def what_if_price(row):
    """Model 1: P(profitable) vs unit price for every category; marks this order."""
    sw = price_sweep(row)
    cur_cat, cur_price = row["category"].iloc[0], float(row["price"].iloc[0])
    fig = go.Figure()
    for i, c in enumerate(CATEGORY_CHOICES):
        d = sw[sw["category"] == c]
        is_cur = c == cur_cat
        fig.add_scatter(x=d["price"], y=d["p_profitable"], mode="lines", name=c,
                        line=dict(width=4 if is_cur else 1.5, color=PALETTE[i % len(PALETTE)]),
                        opacity=1.0 if is_cur else 0.55)
    p_now = float(np.interp(cur_price, sw[sw["category"] == cur_cat]["price"],
                            sw[sw["category"] == cur_cat]["p_profitable"]))
    fig.add_scatter(x=[cur_price], y=[p_now], mode="markers", name="this order",
                    marker=dict(size=14, color="black", symbol="diamond"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(title="What-if: P(Profitable) as the unit price changes (one line per category)",
                      xaxis_title="Unit price (log scale)", yaxis_title="P(Profitable)",
                      xaxis_type="log", yaxis_range=[0, 0.95])
    return fig


def sensitivity_fig(key, row, p_now):
    """Range of probability each input can cause on its own (shows which inputs matter)."""
    pos = LABELS[key][1]
    sv = sensitivity(key, row).iloc[::-1]
    fig = go.Figure(go.Bar(y=sv["input"], x=sv["swing"], base=sv["p_min"], orientation="h",
                           marker_color=INDIGO, opacity=0.85,
                           customdata=np.stack([sv["p_min"], sv["p_max"]], axis=-1),
                           hovertemplate="%{y}: P from %{customdata[0]:.2f} to %{customdata[1]:.2f}<extra></extra>"))
    fig.add_vline(x=p_now, line_dash="dash", line_color="black",
                  annotation_text="this order", annotation_position="top")
    fig.update_layout(title=f"Which inputs change P({pos})? (range when only that input is varied)",
                      xaxis_title=f"P({pos})", xaxis_range=[0, 0.95])
    return fig


def what_if(key, row):
    """Model 2: probability of fast delivery when the region changes."""
    if key == "profit":
        return what_if_price(row)
    neg, pos = LABELS[key]
    col, choices = ("category", CATEGORY_CHOICES) if key == "profit" else ("region", REGION_CHOICES)
    sweep = pd.concat([row.assign(**{col: c}) for c in choices], ignore_index=True)
    res = predict_orders(sweep, tasks=[key])
    d = pd.DataFrame({col.title(): choices, f"P({pos})": res[f"{key}_ann_prob_{pos.lower()}"].values})
    d["Current"] = np.where(d[col.title()] == row[col].iloc[0], "this order", "other")
    fig = px.bar(d, x=col.title(), y=f"P({pos})", color="Current", range_y=[0, 0.95], text_auto=".0%",
                 color_discrete_map={"this order": INDIGO, "other": "#CBD5E1"},
                 title=f"What-if: P({pos}) if the {col} changed")
    return fig


def page_predict(key):
    neg, pos = LABELS[key]
    descriptions = {
    "profit": "profitable or loss-making",
    "delivery": "fast or slower",
    "return": "returned or not returned"
}

    what = descriptions[key]
    hero(f"Predict · {TASK_TITLES[key].split('·')[1].strip()}",
         f"Enter an order to find out whether it will be {what}. This page uses only Model "
         f"{ {'profit': '1', 'delivery': '2', 'return': '3'}[key] }.")
    tab_single, tab_batch = st.tabs(["Single order", "Batch (CSV upload)"])

    with tab_single:
        med = get_data().groupby("category")["price"].median().round(0)
        st.caption("Typical (median) unit price by category: " +
                   " · ".join(f"{c} {int(v)}" for c, v in med.items()) +
                   ". Predictions are most reliable inside these ranges.")
        row = order_form(key)
        if row is not None:
            try:
                res = predict_orders(row, tasks=[key]).iloc[0]
            except Exception as e:  # noqa
                st.error(f"Prediction failed: {e}")
                st.stop()
            p = float(res[f"{key}_ann_prob_{pos.lower()}"])
            verdict = res[f"{key}_ann_prediction"]
            st.caption(f"Computed order total: **{row['total_amount'].iloc[0]:,.2f}**")
            c1, c2 = st.columns([1, 1])
            with c1:
                chart(prob_gauge(p, pos, risk=key == "return"), 300)
            with c2:
                if key == "return":
                    css = "bad" if verdict == pos else "good"
                else:
                    css = "good" if verdict == pos else "bad"
                st.markdown(f'<div class="verdict {css}">{verdict}</div>', unsafe_allow_html=True)
                conf = max(p, 1 - p)
                st.write(f"ANN confidence: **{min(conf, 0.95) * 100:.1f}%**")
                st.caption(f"HistGradientBoosting cross-check: {res[f'{key}_hgb_prediction']} "
                           f"(P={res[f'{key}_hgb_prob_{pos.lower()}']:.3f})")
                if key == "delivery":
                    st.caption("In this dataset delivery speed is driven almost entirely by region, "
                               "so about 75% accuracy is the realistic ceiling.")
            if key != "return":
                chart(what_if(key, row), 340)
            chart(sensitivity_fig(key, row, p), 380)
            st.caption("Inputs with no visible bar do not affect this model: the dataset contains no signal "
                       "in them (e.g. delivery speed depends only on region; profit on price and category).")

    with tab_batch:
        st.write("Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in REQUIRED_NEW_DATA_COLUMNS) +
                 " (optional: `total_amount`).")
        sample = pd.read_csv(config.ROOT / "data" / "sample_new_orders.csv")
        st.download_button("Download sample CSV", sample.to_csv(index=False), "sample_new_orders.csv", "text/csv",
                           key=f"dl_sample_{key}")
        up = st.file_uploader("Upload orders CSV", type="csv", key=f"upload_{key}")
        df_new = pd.read_csv(up) if up else None
        if df_new is None and st.checkbox("Use the sample file instead", key=f"use_sample_{key}"):
            df_new = sample
        if df_new is not None:
            try:
                res = predict_orders(df_new, tasks=[key])
            except Exception as e:  # noqa
                st.error(f"Could not score the file: {e}")
                return
            st.success(f"Scored {len(res):,} orders")
            st.dataframe(res)
            st.download_button("Download predictions", res.to_csv(index=False), f"{key}_predictions.csv",
                               "text/csv", key=f"dl_pred_{key}")
            c1, c2 = st.columns(2)
            with c1:
                chart(px.histogram(res, x=f"{key}_ann_prob_{pos.lower()}", nbins=20,
                                   title=f"P({pos}) distribution"), 340)
            with c2:
                pie_args = {"color_discrete_map": {pos: RED, neg: GREEN}} if key == "return" else {
                    "color_discrete_sequence": [GREEN, RED]
                }
                chart(px.pie(res, names=f"{key}_ann_prediction", hole=0.5, title="Predicted outcome",
                             **pie_args), 340)


# ------------------------------------------------------------------ main
if not artifacts_ready():
    hero("NexusRetail", "Trained models were not found.")
    st.warning("Train them once (takes a few minutes).")
    if st.button("Train models now", type="primary"):
        with st.spinner("Training both ANN models..."):
            out = subprocess.run([sys.executable, str(ROOT / "run_training.py")], cwd=ROOT,
                                 capture_output=True, text=True)
        st.code(out.stdout[-3000:] + out.stderr[-1500:])
        st.cache_data.clear()
        st.rerun()
    st.stop()

_vp = version_problem()
if _vp:
    st.warning(_vp)
    if st.button("Rebuild models now"):
        with st.spinner("Training both ANN models (a few minutes)..."):
            out = subprocess.run([sys.executable, str(ROOT / "run_training.py")], cwd=ROOT,
                                 capture_output=True, text=True)
        st.code(out.stdout[-3000:] + out.stderr[-1500:])
        st.cache_data.clear()
        st.rerun()

PAGES = {
    "Overview": page_overview,
    "Dataset": page_dataset,
    "Data Model": page_data_model,

    "Model 1 - Profit vs Loss": lambda: page_model("profit"),
    "Model 2 - Delivery speed": lambda: page_model("delivery"),
    "Model 3 - Return Risk": lambda: page_model("return"),

    "Predict - Profit / Loss": lambda: page_predict("profit"),
    "Predict - Delivery speed": lambda: page_predict("delivery"),
    "Predict - Return Risk": lambda: page_predict("return"),
}
st.sidebar.markdown("## NexusRetail")
random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=2147483647,
    value=saved_random_seed(),
    step=1,
    help="Retrain with a different seed to regenerate model predictions and evaluation metrics.",
)
if st.sidebar.button("Retrain models with this seed"):
    with st.spinner(f"Training models with random seed {random_seed}..."):
        out = subprocess.run(
            [sys.executable, str(ROOT / "run_training.py"), "--seed", str(random_seed)],
            cwd=ROOT, capture_output=True, text=True,
        )
    st.code(out.stdout[-3000:] + out.stderr[-1500:])
    if out.returncode != 0:
        st.error(f"Training failed with exit code {out.returncode}; the saved models were not refreshed.")
        st.stop()
    clear_model_cache()
    st.cache_data.clear()
    st.rerun()
page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "3 ANN models: Profitability · Delivery Speed · Return Risk"
)
PAGES[page]()
