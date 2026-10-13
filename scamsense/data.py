# 📦 data.py — rendu public SMS datasets ni kalipi, clean chesi, train/val/test split
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import LABELS

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
LABEL_MAP = {"ham": "legit", "spam": "spam", "smishing": "scam"}


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def load_raw(raw_dir=RAW):
    """Mishra & Soni SMS phishing set (ham/spam/smishing) + Indian telecom SMS (ham/spam)."""
    phish = pd.read_csv(raw_dir / "sms_phishing_5971.csv")
    phish = pd.DataFrame({"text": phish.TEXT, "label": phish.LABEL.str.lower().str.strip(), "source": "sms_phishing"})
    india = pd.read_csv(raw_dir / "india_spam_ham.csv")
    india = pd.DataFrame({"text": india.Msg, "label": india.Label.str.lower().str.strip(), "source": "india_sms"})
    return pd.concat([phish, india], ignore_index=True)


def clean(df):
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df.text.map(_norm)
    df["label"] = df.label.map(LABEL_MAP)
    df = df[df.label.notna() & (df.text.str.len() >= 2)]
    key = df.text.str.lower()
    # 🧹 same message rendu labels tho unte confusing — teeseddam; exact duplicates okkate unchali
    conflict = df.groupby(key).label.transform("nunique") > 1
    df = df[~conflict].assign(_k=key[~conflict]).drop_duplicates("_k").drop(columns="_k")
    return df.reset_index(drop=True)


def split(df, seed=42):
    """70 / 15 / 15, label + source wise stratified (test lo Indian messages kuda untayi)."""
    strat = df.label + "|" + df.source
    train, rest = train_test_split(df, test_size=0.30, stratify=strat, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=strat[rest.index], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits(seed=42):
    return split(clean(load_raw()), seed)


def load_stress_test():
    path = RAW.parent / "india_upi_stress_test.csv"
    df = pd.read_csv(path)
    assert set(df.label) <= set(LABELS)
    return df
