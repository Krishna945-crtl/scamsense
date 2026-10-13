"""ScamSense — fine-tuned DistilBERT that sorts SMS into legit / spam / scam."""
LABELS = ["legit", "spam", "scam"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
