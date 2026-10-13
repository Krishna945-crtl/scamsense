# 📝 report.py — results/*.json nunchi README lo results table + comparison chart update cheyyadam
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS, README = ROOT / "results", ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def table(base, ft):
    rows = [("Test set: macro-F1", "test", "macro_f1", "{:.3f}"),
            ("Test set: scam recall", "test", "scam_recall", "{:.0%}"),
            ("Test set: scam precision", "test", "scam_precision", "{:.0%}"),
            ("India UPI stress test: macro-F1", "stress_test", "macro_f1", "{:.3f}"),
            ("India UPI stress test: scam recall", "stress_test", "scam_recall", "{:.0%}"),
            ("Real messages wrongly flagged as scam (test + stress)", None, "legit_flagged_as_scam", "{}")]
    head = "| Metric | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |\n|---|---|---|\n"
    lines = []
    for label, split, key, fmt in rows:
        cell = lambda r: "—" if r is None else fmt.format(
            r["test"][key] + r["stress_test"][key] if split is None else r[split][key])
        lines.append(f"| {label} | {cell(base)} | {'**' + cell(ft) + '**' if ft else '_run fine-tuning_'} |")
    return head + "\n".join(lines)


def chart(base, ft, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = ["Test macro-F1", "Test scam recall", "Stress macro-F1", "Stress scam recall"]
    get = lambda r: [r["test"]["macro_f1"], r["test"]["scam_recall"], r["stress_test"]["macro_f1"], r["stress_test"]["scam_recall"]]
    x, w = np.arange(len(groups)), 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for off, r, name, color in [(-w / 2 - 0.01, base, "TF-IDF + LogReg", "#eb6834"), (w / 2 + 0.01, ft, "Fine-tuned DistilBERT", "#2a78d6")]:
        vals = get(r)
        ax.bar(x + off, vals, w, label=name, color=color)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.12)
    ax.set_title("Baseline vs fine-tuned model", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncols=2)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.yaxis.grid(True, color="#ebeae6")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    base, ft = _load("baseline_metrics.json"), _load("finetuned_metrics.json")
    block = table(base, ft)
    if ft:
        chart(base, ft, RESULTS / "comparison.png")
        block += (f"\n\n![Baseline vs fine-tuned](results/comparison.png)\n\n"
                  f"Fine-tuned for {ft['params']['epochs']} epochs on `{ft['device']}` in {ft['train_minutes']} min "
                  f"(best validation macro-F1 {ft['best_val_macro_f1']:.3f}). "
                  f"Confusion matrices: [test](results/confusion_test.png) · [stress test](results/confusion_stress.png). "
                  f"Every mistake is listed in [errors.csv](results/errors.csv).")
    text = README.read_text(encoding="utf-8")
    text = text[:text.index(START) + len(START)] + "\n" + block + "\n" + text[text.index(END):]
    README.write_text(text, encoding="utf-8")
    print("README results updated")


if __name__ == "__main__":
    main()
