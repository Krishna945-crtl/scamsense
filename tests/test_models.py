import json

import torch

from scamsense import baseline
from scamsense.data import load_splits
from scamsense.finetune import class_weights, make_loader, train
from scamsense.metrics import evaluate


def test_metrics_count_dangerous_mistakes():
    m = evaluate(["legit", "scam", "scam", "spam"], ["scam", "legit", "scam", "spam"])
    assert m["legit_flagged_as_scam"] == 1 and m["scams_missed_as_legit"] == 1
    assert m["scam_recall"] == 0.5


def test_baseline_is_strong_on_the_test_set():
    train_df, _, test = load_splits()
    model = baseline.build().fit(train_df.text, train_df.label)
    assert evaluate(test.label, model.predict(test.text))["macro_f1"] > 0.9


def test_rare_scam_class_gets_more_weight():
    w = class_weights(["legit"] * 90 + ["spam"] * 7 + ["scam"] * 3, "cpu")
    assert w[2] > w[1] > w[0]


def test_training_loop_learns(tiny_tokenizer, tiny_model):
    """Chinna batch ni overfit cheyyagaligithe — loop, loss, optimizer anni correct ga connect ayyayi."""
    train_df, _, _ = load_splits()
    batch_df = train_df.groupby("label").head(8)
    batch = next(iter(make_loader(tiny_tokenizer, batch_df.text, batch_df.label, 48, len(batch_df), False)))
    labels = batch.pop("labels")
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):  # 80 steps — tiny model ki 40 saripovu
        loss = torch.nn.functional.cross_entropy(tiny_model(**batch).logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0]


def test_full_pipeline_writes_results(tiny_tokenizer, tiny_model, tmp_path):
    res = train(epochs=1, batch_size=16, max_len=48, tokenizer=tiny_tokenizer, model=tiny_model,
                out_dir=tmp_path, limit=40, save_model=False)
    saved = json.loads((tmp_path / "finetuned_metrics.json").read_text())
    assert saved["test"]["n"] == res["test"]["n"] > 0
    assert {"confusion_test.png", "confusion_stress.png", "errors.csv"} <= {p.name for p in tmp_path.iterdir()}
