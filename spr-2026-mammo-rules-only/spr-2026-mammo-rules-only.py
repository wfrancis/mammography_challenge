"""Minimal submission - rule specialist only."""
import re
from pathlib import Path
import numpy as np
import pandas as pd

def find_csv(name):
    for p in Path("/kaggle/input").rglob(name):
        return p
    raise FileNotFoundError(name)

train_path = find_csv("train.csv")
test_path = train_path.parent / "test.csv"
print(f"train: {train_path}")
print(f"test: {test_path}")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
labels = sorted(train_df["target"].unique().tolist())
lti = {l: i for i, l in enumerate(labels)}
print(f"train={len(train_df)}  test={len(test_df)}  classes={labels}")

text = test_df["report"].fillna("").astype(str).str.lower()
rule = np.full((len(test_df), len(labels)), 1e-6, dtype=np.float32)
rule[:, lti[2]] = 0.55; rule[:, lti[1]] = 0.15; rule[:, lti[0]] = 0.10
rule[:, lti[3]] = 0.10; rule[:, lti[4]] = 0.05; rule[:, lti[5]] = 0.025; rule[:, lti[6]] = 0.025

path = (text.str.contains("biops", regex=False) | text.str.contains("carcin", regex=False) | text.str.contains("malign", regex=False))
susp5 = (text.str.contains("espicul", regex=False) | text.str.contains("retra", regex=False)) & ~path
calc4 = (text.str.contains("amorf", regex=False) | text.str.contains("pleomorf", regex=False) | text.str.contains("segmentar", regex=False)) & ~path & ~susp5
stable3 = (text.str.contains("estável", regex=False) | text.str.contains("estavel", regex=False) | text.str.contains("controle", regex=False)) & ~path & ~susp5 & ~calc4
recall0 = (text.str.contains("compress", regex=False) | text.str.contains("magnific", regex=False) | text.str.contains("reavalia", regex=False)) & ~path & ~susp5 & ~calc4 & ~stable3
normal1 = text.str.contains("sem alterações significativas", regex=False) & ~path & ~susp5 & ~calc4 & ~stable3 & ~recall0

for mask, label, conf in [(path,6,0.97),(susp5,5,0.93),(calc4,4,0.90),(stable3,3,0.88),(recall0,0,0.85),(normal1,1,0.80)]:
    idx = np.where(mask.to_numpy())[0]
    rule[idx] = 1e-6
    rule[idx, lti[label]] = conf

pred_idx = rule.argmax(axis=1)
pred_labels = [labels[i] for i in pred_idx]
sub = pd.DataFrame({"ID": test_df["ID"], "target": pred_labels})
sub.to_csv("submission.csv", index=False)
print(f"rows={len(sub)}")
print(sub.head())
