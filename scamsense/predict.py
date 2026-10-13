# 🔮 predict.py — python -m scamsense.predict "Your KYC expired, update now..."
import argparse
from pathlib import Path

import torch

from . import LABELS

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "results" / "model"


class ScamDetector:
    def __init__(self, model_dir=DEFAULT_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt")
        probs = torch.softmax(self.model(**enc).logits, dim=-1)
        return [{"label": LABELS[int(p.argmax())], "confidence": round(float(p.max()), 3),
                 "probs": {l: round(float(x), 3) for l, x in zip(LABELS, p)}} for p in probs]


def main():
    p = argparse.ArgumentParser(description="Classify SMS as legit / spam / scam")
    p.add_argument("messages", nargs="+")
    p.add_argument("--model-dir", default=DEFAULT_MODEL)
    a = p.parse_args()
    for msg, r in zip(a.messages, ScamDetector(a.model_dir)(a.messages)):
        icon = {"scam": "🚨", "spam": "📢", "legit": "✅"}[r["label"]]
        print(f"{icon} {r['label'].upper():5s} ({r['confidence']:.0%})  {msg[:80]}")


if __name__ == "__main__":
    main()
