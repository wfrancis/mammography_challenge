"""
Offline Kaggle submission kernel — 3-way blend on hidden test set:
  1. BERTimbau-large (5 folds, weights loaded from attached dataset)
  2. TF-IDF + LR sparse baseline (trained inline on train.csv)
  3. Deterministic rule specialist (pattern matches on text)

Expected attached dataset layout (FLAT, no subdirs):
  mammo-bertimbau-flat/
    blend.json
    fold0_pytorch_model.bin + fold0_config.json + fold0_tokenizer.json + fold0_vocab.txt + ...
    fold1_...
    fold2_...
    fold3_...
    fold4_...

Flat layout avoids Kaggle's zip/unzip corruption on 1GB+ bin files.

No internet, no GPU. Uses torch.quantization.quantize_dynamic for CPU speedup.
Writes submission.csv.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

COMPETITION_NAME = "spr-2026-mammography-report-classification"
DEFAULT_COMPETITION_PATH = Path("/kaggle/input") / COMPETITION_NAME
MAX_LENGTH = 384
BERT_BATCH_SIZE = 4
BACKBONE_FALLBACK = "neuralmind/bert-large-portuguese-cased"

WHITESPACE_RE = re.compile(r"\s+")


def normalize_report(text):
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\n\r", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return " [SEP] ".join(lines)


def resolve_data_root():
    if (DEFAULT_COMPETITION_PATH / "train.csv").exists() and (DEFAULT_COMPETITION_PATH / "test.csv").exists():
        return DEFAULT_COMPETITION_PATH
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_path in kaggle_input.rglob("train.csv"):
            candidate = train_path.parent
            if (candidate / "test.csv").exists():
                return candidate
    raise FileNotFoundError("Could not locate train.csv and test.csv.")


def resolve_weights_root():
    """Find the dataset dir containing flat fold{N}_pytorch_model.bin files."""
    candidates = [
        Path("/kaggle/input/mammo-bert-fold1"),
        Path("/kaggle/input/mammo-bertimbau-flat"),
        Path("/kaggle/input/mammo-bertimbau-large"),
    ]
    for c in candidates:
        if c.exists() and any((c / f"fold{i}_pytorch_model.bin").exists() for i in range(5)):
            return c
    # wider search for any fold*_pytorch_model.bin
    for i in range(5):
        for p in Path("/kaggle/input").rglob(f"fold{i}_pytorch_model.bin"):
            return p.parent
    raise FileNotFoundError("Could not locate fold{N}_pytorch_model.bin under /kaggle/input.")


class ReportDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            return_attention_mask=True,
        )


def predict_logits(model, loader, device):
    model.eval()
    chunks = []
    total = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            chunks.append(outputs.logits.detach().float().cpu().numpy())
            if i % 50 == 0:
                print(f"    batch {i}/{total}", flush=True)
    return np.concatenate(chunks)


def build_fold_dir(flat_root: Path, fold_id: int, work_base: Path) -> Path:
    """Reassemble a real fold directory from flat files (transformers wants
    `from_pretrained(dir)` where the dir has config.json + tokenizer files)."""
    out = work_base / f"fold{fold_id}"
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "special_tokens_map.json",
    ]:
        src = flat_root / f"fold{fold_id}_{name}"
        dst = out / name
        if src.exists() and not dst.exists():
            # symlink to avoid copying 1GB bin files
            try:
                dst.symlink_to(src)
            except Exception:
                shutil.copy(src, dst)
    return out


def train_sparse_predict_test(train_df, test_df, labels):
    train_text = train_df["report"].fillna("").astype(str).map(normalize_report)
    test_text = test_df["report"].fillna("").astype(str).map(normalize_report)
    word = TfidfVectorizer(
        analyzer="word", lowercase=True, strip_accents="unicode",
        ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, strip_accents="unicode",
        ngram_range=(3, 5), min_df=2, max_features=400000, sublinear_tf=True,
    )
    x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)], format="csr")
    x_test = hstack([word.transform(test_text), char.transform(test_text)], format="csr")
    y = train_df["target"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    m = OneVsRestClassifier(
        LogisticRegression(C=3.0, class_weight="balanced", max_iter=2000, solver="liblinear"),
        n_jobs=1,
    )
    m.fit(x_train, y)
    return m.predict_proba(x_test).astype(np.float32)


def rule_specialist_test(test_df, labels):
    text = test_df["report"].fillna("").astype(str).str.lower()
    lti = {l: i for i, l in enumerate(labels)}
    rule = np.full((len(test_df), len(labels)), 1e-6, dtype=np.float32)
    rule[:, lti[2]] = 0.55; rule[:, lti[1]] = 0.15; rule[:, lti[0]] = 0.10
    rule[:, lti[3]] = 0.10; rule[:, lti[4]] = 0.05; rule[:, lti[5]] = 0.025; rule[:, lti[6]] = 0.025
    path = text.str.contains("biops", regex=False) | text.str.contains("carcin", regex=False) | text.str.contains("malign", regex=False)
    susp5 = (text.str.contains("espicul", regex=False) | text.str.contains("retra", regex=False)) & ~path
    calc4 = (text.str.contains("amorf", regex=False) | text.str.contains("pleomorf", regex=False) | text.str.contains("segmentar", regex=False)) & ~path & ~susp5
    stable3 = (text.str.contains("estável", regex=False) | text.str.contains("estavel", regex=False) | text.str.contains("controle", regex=False)) & ~path & ~susp5 & ~calc4
    recall0 = (text.str.contains("compress", regex=False) | text.str.contains("magnific", regex=False) | text.str.contains("reavalia", regex=False)) & ~path & ~susp5 & ~calc4 & ~stable3
    normal1 = text.str.contains("sem alterações significativas", regex=False) & ~path & ~susp5 & ~calc4 & ~stable3 & ~recall0
    for mask, label, conf in [(path,6,0.97),(susp5,5,0.93),(calc4,4,0.90),(stable3,3,0.88),(recall0,0,0.85),(normal1,1,0.80)]:
        idx = np.where(mask.to_numpy())[0]
        rule[idx] = 1e-6
        rule[idx, lti[label]] = conf
        remain = (1.0 - conf) / (len(labels) - 1)
        for c in range(len(labels)):
            if c != lti[label]:
                rule[idx, c] = remain
    return rule


def main():
    data_root = resolve_data_root()
    weights_root = resolve_weights_root()
    print(f"data_root={data_root}\nweights_root={weights_root}", flush=True)

    train_df = pd.read_csv(data_root / "train.csv")
    test_df = pd.read_csv(data_root / "test.csv")
    labels = sorted(train_df["target"].unique().tolist())
    num_classes = len(labels)
    print(f"train={len(train_df)}  test={len(test_df)}  classes={labels}", flush=True)

    # Blend config — hardcoded so no dataset.json round-trip needed
    # Based on 2-fold partial OOF analysis (0.806 on 40% coverage), softened for
    # single-fold submission (less averaging of bert predictions).
    blend_cfg = {
        "weights": {"bertimbau": 0.5, "sparse": 0.35, "rule": 0.15},
        "offsets": [0.0, -0.15, 0.15, 0.0, 0.05, 0.05, -0.05],
    }
    # Optionally override from dataset blend.json if present
    blend_path = weights_root / "blend.json"
    if blend_path.exists():
        blend_cfg = json.loads(blend_path.read_text())
    print(f"blend: {blend_cfg}", flush=True)

    w_bert = float(blend_cfg["weights"].get("bertimbau", 0.0))
    w_sparse = float(blend_cfg["weights"].get("sparse", 0.0))
    w_rule = float(blend_cfg["weights"].get("rule", 0.0))
    offsets = np.asarray(blend_cfg["offsets"], dtype=np.float32)

    # Sparse (always compute, cheap)
    print("\n[sparse] fitting TF-IDF + LR...", flush=True)
    sparse_proba = train_sparse_predict_test(train_df, test_df, labels) if w_sparse > 0 else np.zeros((len(test_df), num_classes), dtype=np.float32)
    print(f"[sparse] done  shape={sparse_proba.shape}", flush=True)

    # Rule (always cheap)
    print("[rule] applying patterns...", flush=True)
    rule_proba = rule_specialist_test(test_df, labels)
    print(f"[rule] done  shape={rule_proba.shape}", flush=True)

    # BERTimbau (5 folds from flat layout)
    bert_proba = np.zeros((len(test_df), num_classes), dtype=np.float32)
    if w_bert > 0:
        fold_ids = [i for i in range(5) if (weights_root / f"fold{i}_pytorch_model.bin").exists()]
        print(f"\n[bert] folds found: {fold_ids}", flush=True)
        assert len(fold_ids) >= 1

        work = Path(tempfile.mkdtemp(prefix="mammo_bert_"))
        # Load tokenizer once from fold0's files
        fold0_dir = build_fold_dir(weights_root, fold_ids[0], work)
        tokenizer = AutoTokenizer.from_pretrained(fold0_dir, use_fast=True)
        test_texts = [normalize_report(t) for t in test_df["report"].fillna("").astype(str).tolist()]
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        test_dataset = ReportDataset(test_texts, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(
            test_dataset, batch_size=BERT_BATCH_SIZE, shuffle=False,
            collate_fn=collator, num_workers=0, pin_memory=False,
        )

        device = torch.device("cpu")
        active = 0
        for fi in fold_ids:
            fd = build_fold_dir(weights_root, fi, work)
            print(f"\n[bert] loading fold{fi}...", flush=True)
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    fd, num_labels=num_classes,
                )
                model.to(device)
                # int8 dynamic quantization for CPU speedup
                try:
                    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    print(f"  fold{fi}: quantized to int8", flush=True)
                except Exception as e:
                    print(f"  fold{fi}: quantization failed ({e}), fp32", flush=True)
                logits = predict_logits(model, test_loader, device)
                proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                bert_proba += proba
                active += 1
                print(f"  fold{fi}: done", flush=True)
                del model
            except Exception as e:
                print(f"  fold{fi}: FAILED to load/infer: {e}  (skipping)", flush=True)
        bert_proba = bert_proba / max(1, active)
        print(f"\n[bert] combined {active} folds", flush=True)

    blended = w_bert * bert_proba + w_sparse * sparse_proba + w_rule * rule_proba
    pred_idx = (blended + offsets[None, :]).argmax(axis=1)
    pred_labels = [labels[i] for i in pred_idx]

    submission = pd.DataFrame({"ID": test_df["ID"], "target": pred_labels})
    submission.to_csv("submission.csv", index=False)
    print(f"\n=== submission ({len(submission)} rows) ===")
    print(submission.head(10))
    print(f"class dist:\n{submission['target'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
