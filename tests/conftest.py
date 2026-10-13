import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

from scamsense import LABEL2ID, LABELS
from scamsense.data import load_splits


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Offline tokenizer (internet lekunda tests run avvali) — mana train messages meeda train chestam."""
    train, _, _ = load_splits()
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.train_from_iterator(train.text, trainers.WordPieceTrainer(
        vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


@pytest.fixture
def tiny_model(tiny_tokenizer):
    # 🎲 seed fix + dropout off — lekapothe random init valla test kokasari fail avthundi
    torch.manual_seed(0)
    cfg = DistilBertConfig(vocab_size=len(tiny_tokenizer), dim=32, hidden_dim=64, n_layers=2, n_heads=2,
                           dropout=0.0, attention_dropout=0.0, seq_classif_dropout=0.0,
                           num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    return DistilBertForSequenceClassification(cfg)
