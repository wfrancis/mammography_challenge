"""
SPR 2026 Mammography Report Classification — PhD v8: 3-kernel argmax vote
==========================================================================
Combines 3 LB-validated kernels via per-row argmax voting with priority tie-break:

  Kernel 1 (LB 0.80972, HIGHEST PRIORITY): multihead_thresh_tuned
    - Tuned thresholds 0.10/0.15/0.23/0.38, binary class-0 detector, safe rules.

  Kernel 2 (LB 0.80591): multihead_80591 (template-safe multihead)
    - Looser thresholds 0.15/0.20/0.25/0.35, exact-match template overrides.

  Kernel 3 (multihead_thresh_mega): cleanlab + 5 binary detectors + extended rules
    - 177 cleanlab label fixes, binary detectors for classes 0/3/4/5/6, expanded guardrails.

To save runtime, we compute the SHARED heavy TF-IDF features once, then derive
3 sets of predictions from differentiated downstream logic.

Vote: rare-aware mode of (p1, p2, p3). Rare classes (4/5/6) are favored when
2+ kernels agree OR when kernel 1 (highest LB) calls a rare class. Otherwise
plurality wins, with kernel 1 breaking ties.

Output: submission.csv with columns ID, target.
"""
from pathlib import Path

def _resolve():
    c = Path("/kaggle/input/spr-2026-mammography-report-classification")
    if (c / "train.csv").exists(): return c
    local = Path("data/raw")
    if (local / "train.csv").exists() and (local / "test.csv").exists(): return local
    for p in Path("/kaggle/input").rglob("train.csv"):
        if (p.parent / "test.csv").exists(): return p.parent
    raise FileNotFoundError("train/test not found")

_D = _resolve()

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import re
import numpy as np
import pandas as pd
import hashlib
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

TRAIN_PATH = str(_D / 'train.csv')
TEST_PATH  = str(_D / 'test.csv')


# ─── CLEANLAB FIXES (from multihead_thresh_mega / multihead_v2) ────────────────
CLEANLAB_FIXES = [
    (7937, 2), (7587, 4), (6741, 2), (3705, 0), (13000, 2), (15197, 4),
    (15677, 2), (4659, 6), (18152, 4), (3650, 0), (7392, 2), (4193, 0),
    (4586, 4), (12523, 2), (7267, 4), (16254, 2), (10700, 4), (3213, 4),
    (15193, 0), (16636, 4), (3757, 4), (15852, 1), (13603, 1), (8799, 2),
    (438, 0), (14535, 1), (3726, 0), (2719, 1), (8108, 0), (16867, 2),
    (17801, 2), (14484, 1), (15778, 4), (1741, 2), (14551, 0), (1752, 0),
    (368, 4), (14621, 1), (2291, 0), (10304, 2), (5898, 0), (15854, 1),
    (7077, 4), (15370, 4), (9293, 4), (16200, 1), (15527, 0), (15700, 0),
    (3546, 0), (4083, 0), (6272, 0), (5909, 0), (17679, 1), (528, 0),
    (3673, 0), (13481, 0), (11080, 4), (11747, 1), (100, 4), (13389, 2),
    (4097, 0), (17619, 4), (8249, 0), (147, 4), (2351, 4), (17203, 1),
    (1310, 4), (17285, 0), (4383, 0), (7911, 0), (11573, 0), (4942, 0),
    (5945, 0), (10272, 0), (470, 0), (7803, 1), (4119, 4), (4595, 0),
    (3854, 4), (3982, 0), (1010, 6), (2616, 0), (961, 0), (6339, 2),
    (7516, 0), (4739, 6), (1846, 4), (2574, 0), (5023, 4), (14935, 4),
    (3980, 2), (8452, 0), (14802, 1), (10862, 4), (3286, 4), (10534, 0),
    (12232, 0), (12476, 0), (9836, 0), (14687, 0), (6900, 4), (5840, 4),
    (293, 4), (15209, 2), (6430, 6), (5500, 0), (2531, 0), (6066, 0),
    (3973, 0), (6277, 0), (18083, 4), (1168, 0), (7703, 5), (13866, 0),
    (15650, 0), (5915, 0), (10444, 0), (16810, 4), (5870, 0), (501, 0),
    (1549, 0), (8556, 4), (8907, 2), (201, 4), (12488, 4), (16974, 4),
    (9526, 0), (5503, 4), (8622, 0), (15640, 0), (10437, 5), (8241, 4),
    (8378, 4), (9691, 4), (12789, 5), (11450, 0), (2410, 0), (8123, 0),
    (12245, 0), (16773, 5), (4988, 0), (7008, 0), (15647, 0), (237, 0),
    (1355, 0), (33, 0), (15138, 0), (4006, 0), (7126, 5), (10717, 0),
    (10337, 4), (10455, 4), (8202, 4), (3595, 0), (5346, 0), (5941, 0),
    (2454, 0), (315, 0), (5439, 4), (12475, 0), (11537, 0), (17650, 0),
    (7621, 0), (13712, 5), (6268, 5), (17421, 0), (16721, 4), (9083, 5),
    (11178, 5), (2541, 0), (18054, 4), (8926, 4), (12275, 4), (9041, 5),
    (10691, 5), (10141, 5), (6987, 4),
]


# ─── PREPROCESSING ─────────────────────────────────────────────────────────────
def clean_achados(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    match = re.search(r'achados:(.*?)(análise comparativa:|$)', s, flags=re.DOTALL)
    if match:
        s = match.group(1).strip()
    s = re.sub(r'[0-9]+,[0-9]+', 'NUM', s)
    s = re.sub(r'[0-9]+', 'NUM', s)
    return s


def clean_full(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r'[0-9]+,[0-9]+', 'NUM', s)
    s = re.sub(r'[0-9]+', 'NUM', s)
    return s


def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def extract_dense_features(df: pd.DataFrame) -> csr_matrix:
    text = df['report'].fillna('').astype(str).str.lower()
    features = pd.DataFrame({
        'report_length':   text.apply(len),
        'has_measurement': text.str.contains(r'\b(?:cm|mm|medindo)\b', regex=True).astype(int),
        'has_spiculation': text.str.contains(r'espiculad', regex=True).astype(int),
        'has_distortion':  text.str.contains(r'distorção arquitetural', regex=True).astype(int),
        'has_biopsy':      text.str.contains(r'biopsy|biópsia|resultado de cine|carcinoma', regex=True).astype(int),
    })
    return csr_matrix(features.values)


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
print(f"Data root: {_D}")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f'Train: {train.shape}, Test: {test.shape}')

train["achados"] = train["report"].apply(clean_achados)
train["full"]    = train["report"].apply(clean_full)
test["achados"]  = test["report"].apply(clean_achados)
test["full"]     = test["report"].apply(clean_full)
train["group"]   = train["report"].apply(stable_hash)

y_orig = train["target"].astype(int).values
print(f'Target distribution:\n{pd.Series(y_orig).value_counts().sort_index()}')

# y_clean: cleanlab-corrected labels (used by kernel 3)
y_clean = y_orig.copy()
n_fixed = 0
for idx, to in CLEANLAB_FIXES:
    if idx < len(y_clean):
        y_clean[idx] = to
        n_fixed += 1
print(f"Applied {n_fixed} cleanlab fixes (used only by kernel 3)")


# ─── SHARED TF-IDF + DENSE FEATURES (computed once, reused by all 3 kernels) ──
print("\n=== Building shared TF-IDF features ===")
X_train_dense = extract_dense_features(train)
X_test_dense  = extract_dense_features(test)

tfidf_A = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.95,
                             sublinear_tf=True, max_features=80000))
])
X_train_A = tfidf_A.fit_transform(train["achados"])
X_test_A  = tfidf_A.transform(test["achados"])

tfidf_F = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.95,
                             sublinear_tf=True, max_features=80000))
])
X_train_F = tfidf_F.fit_transform(train["full"])
X_test_F  = tfidf_F.transform(test["full"])

tfidf_F2 = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=3, max_df=0.95,
                             sublinear_tf=True, max_features=100000))
])
X_train_F2 = tfidf_F2.fit_transform(train["full"])
X_test_F2  = tfidf_F2.transform(test["full"])

X_train_lgb = hstack([X_train_A, X_train_dense]).tocsr()
X_test_lgb  = hstack([X_test_A,  X_test_dense]).tocsr()

print(f"SVC-A: {X_train_A.shape[1]:,}  SVC-F: {X_train_F.shape[1]:,}  "
      f"SVC-F2: {X_train_F2.shape[1]:,}  LGB: {X_train_lgb.shape[1]:,}")


# ─── HELPER: train the shared base ensemble on a label vector ─────────────────
def train_base_ensemble(y_labels, tag=""):
    """Trains SVC-A, SVC-F, SVC-F2, LGB and returns the blended ensemble proba."""
    print(f"\n[{tag}] Training SVC-A...")
    svc_A = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
        cv=3, method='sigmoid')
    svc_A.fit(X_train_A, y_labels)
    proba_A = svc_A.predict_proba(X_test_A)

    print(f"[{tag}] Training SVC-F...")
    svc_F = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
        cv=3, method='sigmoid')
    svc_F.fit(X_train_F, y_labels)
    proba_F = svc_F.predict_proba(X_test_F)

    print(f"[{tag}] Training SVC-F2...")
    svc_F2 = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
        cv=3, method='sigmoid')
    svc_F2.fit(X_train_F2, y_labels)
    proba_F2 = svc_F2.predict_proba(X_test_F2)

    print(f"[{tag}] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(class_weight='balanced', n_estimators=300,
                                   learning_rate=0.05, max_depth=6,
                                   random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train_lgb, y_labels)
    proba_lgb = lgb_model.predict_proba(X_test_lgb)

    svc_ensemble   = 0.25 * proba_A + 0.40 * proba_F + 0.35 * proba_F2
    ensemble_proba = 0.70 * svc_ensemble + 0.30 * proba_lgb
    return ensemble_proba, svc_F


def train_binary_detector(X_tr, X_te, y_labels, target_cls, background_cls=2):
    mask = np.isin(y_labels, [target_cls, background_cls])
    if mask.sum() < 20:
        return np.zeros(X_te.shape[0], dtype=np.float32)
    y_bin = (y_labels[mask] == target_cls).astype(int)
    svc = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
        cv=3, method='sigmoid')
    svc.fit(X_tr[mask], y_bin)
    return svc.predict_proba(X_te)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 1: multihead_thresh_tuned (LB 0.80972, HIGHEST PRIORITY)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("KERNEL 1: multihead_thresh_tuned (LB 0.80972)")
print("="*80)
ensemble_1, svc_F_1 = train_base_ensemble(y_orig, tag="K1")
# Binary class-0 detector on {0, 2}
print("[K1] Training binary class-0 detector...")
mask_02 = np.isin(y_orig, [0, 2])
y_02    = (y_orig[mask_02] == 0).astype(int)
bin0_svc_1 = CalibratedClassifierCV(
    LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
    cv=3, method='sigmoid')
bin0_svc_1.fit(X_train_F[mask_02], y_02)
bin0_test_proba_1 = bin0_svc_1.predict_proba(X_test_F)[:, 1]

preds_1 = np.argmax(ensemble_1, axis=1).copy()
# Tuned thresholds (+0.01867 OOF lift)
preds_1[(ensemble_1[:, 6] > 0.10)] = 6
preds_1[(ensemble_1[:, 5] > 0.15) & (preds_1 != 6)] = 5
preds_1[(ensemble_1[:, 4] > 0.23) & (preds_1 != 6) & (preds_1 != 5)] = 4
preds_1[(ensemble_1[:, 3] > 0.38) & (preds_1 != 6) & (preds_1 != 5) & (preds_1 != 4)] = 3
preds_1[(bin0_test_proba_1 > 0.55) & (preds_1 == 2)] = 0


def k1_safe_rules(row, pred):
    text = str(row['report']).lower()
    if re.search(r'resultado de cine grau 3|carcinoma|\bcdis\b', text):
        return 6
    if 'espiculad' in text and 'distorção' in text and pred < 4:
        return 5
    return pred


preds_1_final = np.array(
    [k1_safe_rules(row, int(preds_1[i])) for i, row in test.reset_index(drop=True).iterrows()],
    dtype=int)
print(f"[K1] preds dist: {pd.Series(preds_1_final).value_counts().sort_index().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 2: multihead_80591 (LB 0.80591, template-safe)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("KERNEL 2: multihead_80591 (LB 0.80591) — reuses K1 base ensemble")
print("="*80)
# Reuse K1's ensemble (same y_orig labels) — only thresholds + template overrides differ
ensemble_2 = ensemble_1
bin0_test_proba_2 = bin0_test_proba_1

preds_2 = np.argmax(ensemble_2, axis=1).copy()
# Looser thresholds (LB 0.80591 settings)
preds_2[(ensemble_2[:, 6] > 0.15)] = 6
preds_2[(ensemble_2[:, 5] > 0.20) & (preds_2 != 6)] = 5
preds_2[(ensemble_2[:, 4] > 0.25) & (preds_2 != 6) & (preds_2 != 5)] = 4
preds_2[(ensemble_2[:, 3] > 0.35) & (preds_2 != 6) & (preds_2 != 5) & (preds_2 != 4)] = 3
preds_2[(bin0_test_proba_2 > 0.55) & (preds_2 == 2)] = 0


def k2_safe_rules(row, pred):
    text = str(row['report']).lower()
    if re.search(r'resultado de cine grau 3|carcinoma|\bcdis\b', text):
        return 6
    if 'espiculad' in text and 'distorção' in text and pred < 4:
        return 5
    return pred


# Apply safe rules
preds_2_after_rules = np.array(
    [k2_safe_rules(row, int(preds_2[i])) for i, row in test.reset_index(drop=True).iterrows()],
    dtype=int)


# Template overrides (exact-match): high-purity train report → test override
def _exact_key(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s


def _build_majority_lookup(train_df, key_fn, min_count, min_purity):
    keys = train_df["report"].apply(key_fn)
    frame = pd.DataFrame({"key": keys, "target": train_df["target"].astype(int)})
    frame = frame[frame["key"].astype(str).str.len() > 0]
    counts = frame.groupby(["key", "target"]).size().reset_index(name="n")
    totals = counts.groupby("key")["n"].sum().rename("total")
    best_idx = counts.groupby("key")["n"].idxmax()
    best = counts.loc[best_idx].join(totals, on="key")
    best["purity"] = best["n"] / best["total"]
    best = best[(best["total"] >= min_count) & (best["purity"] >= min_purity)]
    return dict(zip(best["key"], best["target"].astype(int)))


print("[K2] Building exact-template lookup...")
exact_lookup = _build_majority_lookup(train, _exact_key, min_count=1, min_purity=1.0)
preds_2_final = preds_2_after_rules.copy()
template_hits = 0
template_changes = 0
for i, report in enumerate(test["report"].values):
    label = exact_lookup.get(_exact_key(report))
    if label is None:
        continue
    template_hits += 1
    if preds_2_final[i] != label:
        template_changes += 1
        preds_2_final[i] = label
print(f"[K2] Template overrides: hits={template_hits} changes={template_changes}")
print(f"[K2] preds dist: {pd.Series(preds_2_final).value_counts().sort_index().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 3: multihead_thresh_mega (cleanlab + 5 binary detectors + extended rules)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("KERNEL 3: multihead_thresh_mega-style (cleanlab + 5 binary detectors)")
print("="*80)
ensemble_3, svc_F_3 = train_base_ensemble(y_clean, tag="K3")

print("[K3] Training 5 binary detectors (classes 0,3,4,5,6 vs 2)...")
p_bin0_3 = train_binary_detector(X_train_F, X_test_F, y_clean, target_cls=0)
p_bin6_3 = train_binary_detector(X_train_F, X_test_F, y_clean, target_cls=6)
p_bin5_3 = train_binary_detector(X_train_F, X_test_F, y_clean, target_cls=5)
p_bin4_3 = train_binary_detector(X_train_F, X_test_F, y_clean, target_cls=4)
p_bin3_3 = train_binary_detector(X_train_F, X_test_F, y_clean, target_cls=3)

preds_3 = np.argmax(ensemble_3, axis=1).copy()
# Tuned thresholds (same as K1 since labels are cleaner)
preds_3[(ensemble_3[:, 6] > 0.10)] = 6
preds_3[(ensemble_3[:, 5] > 0.15) & (preds_3 != 6)] = 5
preds_3[(ensemble_3[:, 4] > 0.23) & (preds_3 != 6) & (preds_3 != 5)] = 4
preds_3[(ensemble_3[:, 3] > 0.38) & (preds_3 != 6) & (preds_3 != 5) & (preds_3 != 4)] = 3
# 5 binary detector overrides for class 2 → rare classes
preds_3[(p_bin0_3 > 0.60) & (preds_3 == 2)] = 0
preds_3[(p_bin6_3 > 0.40) & (preds_3 == 2)] = 6
preds_3[(p_bin5_3 > 0.60) & (preds_3 == 2)] = 5
preds_3[(p_bin4_3 > 0.70) & (preds_3 == 2)] = 4
preds_3[(p_bin3_3 > 0.65) & (preds_3 == 2)] = 3


def k3_safe_rules(row, pred):
    text = str(row['report']).lower()
    # CLASS 6 — confirmed malignancy
    if re.search(r'resultado de cine grau 3|\bcdis\b|carcinoma invasor|carcinoma ductal invasivo|carcinoma lobular invasivo|neoplasia maligna|recidiva de carcinoma|carcinoma mam[áa]rio', text):
        return 6
    if 'carcinoma' in text and re.search(r'biopsia|biópsia|histopatológico', text):
        return 6
    # CLASS 5 — highly suspicious
    if pred < 4:
        if 'espiculad' in text and 'distorção' in text: return 5
        if 'altamente suspeit' in text: return 5
        if 'categoria 5' in text or 'bi-rads 5' in text: return 5
    # CLASS 4 — suspicious (PMI: pleomorphic + small nodules ⇒ class 4)
    if pred in (1, 2, 3):
        if re.search(r'pequenos n[óo]dulos (com )?calcifica[çc][õo]es pleom[óo]rficas', text): return 4
    # CLASS 3 — probably benign (puntiformes monomórficas)
    if pred == 2:
        if re.search(r'(puntiformes agrupadas no)|(calcifica[çc][õo]es puntiformes e monom[óo]rficas)', text): return 3
    return pred


preds_3_final = np.array(
    [k3_safe_rules(row, int(preds_3[i])) for i, row in test.reset_index(drop=True).iterrows()],
    dtype=int)
print(f"[K3] preds dist: {pd.Series(preds_3_final).value_counts().sort_index().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════════════
# RARE-AWARE ARGMAX VOTE (with kernel 1 priority tie-break)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("VOTING: rare-aware argmax with kernel-1 priority")
print("="*80)


def vote_rare_aware(p1, p2, p3):
    """Per-row vote.

    Rare classes (4, 5, 6) dominate macro-F1 — bias toward keeping them
    when there's any signal, while still using mode-with-priority for the
    common-class majority.

    Logic:
      1. If 2+ kernels agree on a rare class (4/5/6) → that rare class.
      2. If exactly 1 kernel says rare AND it's kernel 1 (highest LB) → kernel 1.
      3. Otherwise: mode of all 3, with kernel 1 winning ties.
    """
    final = []
    for i in range(len(p1)):
        v1, v2, v3 = int(p1[i]), int(p2[i]), int(p3[i])
        votes = [v1, v2, v3]
        rare_votes = [v for v in votes if v in (4, 5, 6)]
        # Rule 1: 2+ kernels agree on rare → keep rare
        if len(rare_votes) >= 2:
            final.append(Counter(rare_votes).most_common(1)[0][0])
            continue
        # Rule 2: exactly 1 rare vote AND it's kernel 1 (highest LB) → trust K1
        if len(rare_votes) == 1 and v1 in (4, 5, 6):
            final.append(v1)
            continue
        # Rule 3: mode with kernel-1 priority tie-break
        c = Counter(votes)
        max_count = max(c.values())
        candidates = [v for v, cnt in c.items() if cnt == max_count]
        if v1 in candidates:
            final.append(v1)
        elif v2 in candidates:
            final.append(v2)
        else:
            final.append(v3)
    return np.array(final, dtype=int)


final_preds = vote_rare_aware(preds_1_final, preds_2_final, preds_3_final)

# Diagnostics
n_unanimous = int(np.sum((preds_1_final == preds_2_final) & (preds_2_final == preds_3_final)))
n_k1_only = int(np.sum((preds_1_final != preds_2_final) & (preds_1_final != preds_3_final)))
n_changed_from_k1 = int(np.sum(final_preds != preds_1_final))
print(f"Unanimous rows: {n_unanimous}/{len(final_preds)}")
print(f"Rows where K1 is solo: {n_k1_only}")
print(f"Rows where final differs from K1: {n_changed_from_k1}")


# ─── SAVE SUBMISSION ───────────────────────────────────────────────────────────
test["target"] = final_preds
submission = test[['ID', 'target']].copy()
submission.to_csv('submission.csv', index=False)

print('\nFinal prediction distribution:')
print(submission['target'].value_counts().sort_index())
print(f'\nSubmission saved to submission.csv ({len(submission)} rows)')
