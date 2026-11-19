"""Static matplotlib figures saved to reports/figures (the dashboard draws interactive ones)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def make_figures(key, m, y_test, proba, loss_curve, val_curve):
    out = config.FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    labels = [m["negative_label"], m["positive_label"]]
    prim = m["models"]["ANN (selected features)"]

    fig, ax = plt.subplots(figsize=(4.6, 4))
    cm = np.array(prim["confusion_matrix"])
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{m['title']} - Confusion matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    fig.tight_layout(); fig.savefig(out / f"{key}_confusion_matrix.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, r in m["models"].items():
        ax[0].plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{name} (AUC {r['roc_auc']:.3f})")
        ax[1].plot(r["pr"]["recall"], r["pr"]["precision"], label=f"{name} (AP {r['avg_precision']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"{key}_roc_pr.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_curve, label="Training loss")
    ax2 = ax.twinx(); ax2.plot(val_curve, color="tab:orange", label="Validation accuracy"); ax2.grid(False)
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{m['title']} - ANN training curve")
    ax2.set_ylabel("Validation accuracy")
    fig.tight_layout(); fig.savefig(out / f"{key}_training_curve.png"); plt.close(fig)

    rk = m["feature_ranking"][:12][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cols = ["tab:green" if r["feature"] in m["selected_features"] else "lightgray" for r in rk]
    ax.barh([r["feature"] for r in rk], [r["importance"] for r in rk], color=cols)
    ax.set(title=f"{m['title']} - HGB permutation importance (green = selected)", xlabel="Importance (log-loss increase)")
    fig.tight_layout(); fig.savefig(out / f"{key}_feature_importance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c["k"] for c in m["selection_curve"]], [c["accuracy"] for c in m["selection_curve"]], "o-")
    ax.axvline(len(m["selected_features"]), color="tab:red", ls="--", label="chosen k")
    ax.set(xlabel="Number of top-ranked features", ylabel="Validation accuracy (HGB)", title=f"{m['title']} - feature count vs accuracy")
    ax.legend(); fig.tight_layout(); fig.savefig(out / f"{key}_selection_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(m["models"]); x = np.arange(len(names)); w = 0.2
    for i, met in enumerate(["accuracy", "precision", "recall", "f1"]):
        ax.bar(x + (i - 1.5) * w, [m["models"][n][met] for n in names], w, label=met)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set(ylim=(0, 1.05), title=f"{m['title']} - model comparison (test set)"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"{key}_model_comparison.png"); plt.close(fig)
