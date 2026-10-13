from scamsense import LABELS
from scamsense.data import clean, load_raw, load_splits, load_stress_test


def test_three_classes_and_no_duplicates():
    df = clean(load_raw())
    assert set(df.label) == set(LABELS)
    assert not df.text.str.lower().duplicated().any()


def test_no_message_leaks_between_splits():
    train, val, test = load_splits()
    seen = set(train.text.str.lower())
    assert not seen & set(val.text.str.lower())
    assert not seen & set(test.text.str.lower())
    assert len(train) > 4 * len(test)


def test_splits_keep_class_balance():
    train, _, test = load_splits()
    share = lambda d: d.label.value_counts(normalize=True)
    assert (share(train) - share(test)).abs().max() < 0.01


def test_stress_test_is_separate_and_labelled():
    stress, (train, _, _) = load_stress_test(), load_splits()
    assert set(stress.label) == set(LABELS) and len(stress) == 60
    assert not set(stress.text) & set(train.text)
