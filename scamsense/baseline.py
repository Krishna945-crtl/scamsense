# 📐 baseline.py — TF-IDF + Logistic Regression. Fine-tuned model deeni kanna better ga undali, lekapothe waste
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from .data import load_splits, load_stress_test
from .metrics import evaluate, save

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build():
    feats = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return make_pipeline(feats, LogisticRegression(max_iter=2000, class_weight="balanced", C=5))


def main():
    train, val, test = load_splits()
    model = build().fit(train.text, train.label)
    stress = load_stress_test()
    india = test[test.source == "india_sms"]
    out = {
        "model": "TF-IDF (word 1-2 + char 3-5 grams) + Logistic Regression",
        "val": evaluate(val.label, model.predict(val.text), "validation"),
        "test": evaluate(test.label, model.predict(test.text), "test"),
        "test_india_subset": evaluate(india.label, model.predict(india.text), "test (Indian SMS only)"),
        "stress_test": evaluate(stress.label, model.predict(stress.text), "India UPI stress test"),
    }
    save(out, RESULTS / "baseline_metrics.json")
    for k in ("test", "stress_test"):
        r = out[k]
        print(f"{r['name']:>22}: macro-F1 {r['macro_f1']:.3f} | scam recall {r['scam_recall']:.2f} | "
              f"legit→scam {r['legit_flagged_as_scam']}")
    return out


if __name__ == "__main__":
    main()
