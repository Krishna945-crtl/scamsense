# 🚨 ScamSense: Fine-tuned DistilBERT for SMS Scam Detection

Sorts SMS messages into **legit**, **spam** (promotions) or **scam** (phishing / fraud) by
**fine-tuning DistilBERT with a hand-written PyTorch training loop**, then checks it on Indian UPI scams
it never saw during training.

Why three classes? A bank OTP, a Diwali sale message and a fake "KYC expired" SMS all look urgent and
mention money. Blocking a real bank alert is almost as bad as letting a scam through, so the model has to
tell all three apart, not just "spam vs not spam".

## Results

<!-- RESULTS:START -->
| Metric | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |
|---|---|---|
| Test set: macro-F1 | 0.944 | _run fine-tuning_ |
| Test set: scam recall | 89% | _run fine-tuning_ |
| Test set: scam precision | 91% | _run fine-tuning_ |
| India UPI stress test: macro-F1 | 0.564 | _run fine-tuning_ |
| India UPI stress test: scam recall | 36% | _run fine-tuning_ |
| Real messages wrongly flagged as scam (test + stress) | 4 | _run fine-tuning_ |
<!-- RESULTS:END -->

**What the numbers say**
- On the public test set, a well-tuned TF-IDF baseline is already strong. That is why the baseline exists: the fine-tuned model has to beat it, not a strawman.
- The real test is the **India UPI stress test**: 60 hand-written messages in Indian formats (KYC expiry, UPI collect-request refunds, "digital arrest", task-job scams, plus real-format OTP and debit alerts) that look nothing like the mostly UK training data. Here the baseline mostly confuses scams with promotions, because it only matches words, not meaning.

## How it works

| Step | Detail |
|---|---|
| **Data** | 7,842 unique messages from 2 public datasets (below): 6,143 legit, 1,171 spam, 528 scam. Duplicates and messages with conflicting labels removed; stratified 70 / 15 / 15 split with zero overlap. |
| **Baseline** | TF-IDF on word 1–2-grams + character 3–5-grams, Logistic Regression with balanced class weights. |
| **Fine-tuning** | `distilbert-base-uncased` + classification head, plain PyTorch loop: AdamW, linear warm-up/decay, gradient clipping, mixed precision on GPU, dynamic padding, best epoch picked on validation macro-F1. |
| **Class imbalance** | Scam is only ~7% of the data, so the loss uses square-root inverse-frequency class weights. |
| **Evaluation** | Macro-F1, scam precision/recall, and the two costly mistakes counted separately: *real message flagged as scam* and *scam missed as legit*. Every error is written to `results/errors.csv` for review. |

## Run it

**Easiest: Google Colab (free GPU, about 5 minutes).** Open `notebooks/train_on_colab.ipynb`, choose *Runtime → Change runtime type → T4 GPU*, then *Run all*.

**Locally:**
```bash
pip install -r requirements.txt
python -m scamsense.baseline                # ~30 s
python -m scamsense.finetune --epochs 3     # GPU ~3 min, laptop CPU ~25 min
python -m scamsense.report                  # refreshes the results table above
python -m scamsense.predict "Your KYC has expired, update PAN now or account will be blocked"
pytest -q                                   # 9 tests, run offline with a tiny random model
```

## Data and credits

- **SMS Phishing Dataset** by Sandhya Mishra and Devpriya Soni (Jaypee Institute of Information Technology), Mendeley Data, 2022: 5,971 messages labelled ham / spam / smishing. Please cite *"SMS Phishing Dataset for Machine Learning and Pattern Recognition"* if you use it.
- **Indian Telecom SMS Spam Collection** by JuniorAlive (MIT licence): 2,267 Indian messages labelled ham / spam.
- **India UPI stress test** (`data/india_upi_stress_test.csv`): 60 messages I wrote to mimic common Indian scam and alert formats. Phone numbers and links are made up. It is used **only for evaluation**, never for training.

## Limitations

- Most scam examples in the training data are UK-style. Indian scams appear mainly in the stress test, which is small (60 messages), so treat those numbers as a direction, not a guarantee.
- Hindi, Telugu and Hinglish messages are barely represented; a multilingual model would be the next step.
- The model sees only the text. It does not check links, sender IDs or UPI handles, which a production filter would also use.

## Project structure

```
scamsense/
  data.py       load, clean, de-duplicate, stratified split
  baseline.py   TF-IDF + Logistic Regression
  finetune.py   PyTorch fine-tuning loop for DistilBERT
  metrics.py    macro-F1, scam recall, costly-mistake counts, confusion matrix plot
  predict.py    command-line classifier
  report.py     writes the results table and chart into this README
notebooks/train_on_colab.ipynb
tests/          data leakage, metrics, baseline, and a training-loop check
```
