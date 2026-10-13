# 📏 metrics.py — model ni ela judge chestam: macro-F1 + scam recall (scam miss avvadam worst case)
import json

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import LABELS


def evaluate(y_true, y_pred, name=""):
    y_true, y_pred = list(y_true), list(y_pred)
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    legit_idx, scam_idx = LABELS.index("legit"), LABELS.index("scam")
    return {
        "name": name,
        "n": len(y_true),
        "accuracy": round(rep["accuracy"], 4),
        # macro-F1 only over classes that actually exist in this set
        "macro_f1": round(f1_score(y_true, y_pred, labels=[l for l in LABELS if l in set(y_true)],
                                   average="macro", zero_division=0), 4),
        "scam_precision": round(rep["scam"]["precision"], 4),
        "scam_recall": round(rep["scam"]["recall"], 4),
        "scam_f1": round(rep["scam"]["f1-score"], 4),
        # 🚨 real messages (OTP, debit alert) ni scam ani block cheyyadam kuda problem
        "legit_flagged_as_scam": int(cm[legit_idx, scam_idx]),
        "scams_missed_as_legit": int(cm[scam_idx, legit_idx]),
        "confusion_matrix": cm.tolist(),
        "per_class": {l: {k: round(rep[l][k], 4) for k in ("precision", "recall", "f1-score", "support")} for l in LABELS},
    }


def save(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(cm, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="#ffffff" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted", color="#52514e")
    ax.set_ylabel("Actual", color="#52514e")
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.tick_params(length=0, colors="#52514e")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
