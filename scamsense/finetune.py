# 🔥 finetune.py — DistilBERT ni plain PyTorch loop tho fine-tune cheyyadam (Trainer vadaledu, prathi step manake telusu)
# Colab GPU lo ~3 min, laptop CPU lo ~20–30 min:   python -m scamsense.finetune --epochs 3

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import LABEL2ID, LABELS
from .data import load_splits, load_stress_test
from .metrics import evaluate, plot_confusion, save

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tok, texts, labels, max_len, batch_size, shuffle):
    enc = tok(list(texts), truncation=True, max_length=max_len)
    items = [{"input_ids": ids, "attention_mask": m, "labels": LABEL2ID[y]}
             for ids, m, y in zip(enc["input_ids"], enc["attention_mask"], labels)]
    # 🧩 dynamic padding — prathi batch ni daani longest message varake pad, waste compute undadu
    collate = lambda batch: tok.pad(batch, return_tensors="pt")
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        probs.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu())
    probs = torch.cat(probs).numpy()
    return [LABELS[i] for i in probs.argmax(1)], probs


def class_weights(labels, device):
    """Scam messages chala takkuva (7%) — loss lo vaatiki ekkuva weight (inverse frequency, sqrt tho soften)."""
    counts = pd.Series(labels).value_counts().reindex(LABELS).fillna(1).to_numpy(dtype=float)
    w = np.sqrt(counts.sum() / counts)
    return torch.tensor(w / w.mean(), dtype=torch.float, device=device)


def train(model_name="distilbert-base-uncased", epochs=3, lr=3e-5, batch_size=16, max_len=128,
          seed=42, weighted_loss=True, out_dir=RESULTS, tokenizer=None, model=None, limit=None, save_model=True):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(seed)
    if limit:   # tests kosam chinna subset
        train_df, val_df, test_df = (d.groupby("label").head(limit).reset_index(drop=True) for d in (train_df, val_df, test_df))
    stress = load_stress_test()

    tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
    model = model or AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    model.to(device)

    train_dl = make_loader(tok, train_df.text, train_df.label, max_len, batch_size, shuffle=True)
    val_dl = make_loader(tok, val_df.text, val_df.label, max_len, 64, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_df.label, device) if weighted_loss else None)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1, best_state = [], -1.0, None
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), labels)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # exploding gradients aapadam
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()

        val_pred, _ = predict(model, val_dl, device)
        val_f1 = evaluate(val_df.label, val_pred)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(running / len(train_dl), 4), "val_macro_f1": val_f1})
        print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.4f} | val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:   # 🏆 best epoch weights matrame unchutam (overfit ayithe last epoch kaadu)
            best_f1, best_state = val_f1, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    minutes = round((time.time() - start) / 60, 1)

    def run(df, name):
        preds, probs = predict(model, make_loader(tok, df.text, df.label, max_len, 64, False), device)
        return evaluate(df.label, preds, name), preds, probs

    test_m, test_pred, _ = run(test_df, "test")
    india = test_df[test_df.source == "india_sms"]
    india_m, _, _ = run(india, "test (Indian SMS only)") if len(india) else ({}, None, None)
    stress_m, stress_pred, stress_prob = run(stress, "India UPI stress test")

    results = {
        "model": model_name, "device": str(device), "train_minutes": minutes,
        "params": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "max_len": max_len,
                   "weighted_loss": weighted_loss, "seed": seed},
        "data": {"train": len(train_df), "val": len(val_df), "test": len(test_df), "stress": len(stress)},
        "history": history, "best_val_macro_f1": best_f1,
        "test": test_m, "test_india_subset": india_m, "stress_test": stress_m,
    }
    save(results, out_dir / "finetuned_metrics.json")
    plot_confusion(test_m["confusion_matrix"], "Fine-tuned DistilBERT — test set", out_dir / "confusion_test.png")
    plot_confusion(stress_m["confusion_matrix"], "Fine-tuned DistilBERT — India UPI stress test",
                   out_dir / "confusion_stress.png")

    # 🔍 error analysis — model ekkada tappu chesindo chudakapothe improve cheyyalem
    errs = pd.concat([
        test_df.assign(pred=test_pred, set="test")[lambda d: d.pred != d.label][["set", "text", "label", "pred"]],
        stress.assign(pred=stress_pred, set="stress", confidence=stress_prob.max(1).round(3))
        [lambda d: d.pred != d.label][["set", "text", "label", "pred", "confidence"]],
    ])
    errs.to_csv(out_dir / "errors.csv", index=False)

    if save_model:
        model.save_pretrained(out_dir / "model")
        tok.save_pretrained(out_dir / "model")
    print(f"✅ test macro-F1 {test_m['macro_f1']:.3f} | stress scam recall {stress_m['scam_recall']:.2f} "
          f"| {minutes} min on {device}")
    return results


def main():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT for SMS scam detection")
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-weighted-loss", action="store_true")
    a = p.parse_args()
    train(a.model, a.epochs, a.lr, a.batch_size, a.max_len, a.seed, not a.no_weighted_loss)


if __name__ == "__main__":
    main()
