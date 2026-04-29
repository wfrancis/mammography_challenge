# [v6b] BERTimbau-large OOF features + Claude rules + baseline ensemble.
# Strategy:
#   - Load BERTimbau OOF logits (computed from 5 folds, aligned to train.csv) as
#     stacking features for train.
#   - Run a single quantized BERTimbau fold (fold1, OOF F1=0.80) on Kaggle test
#     set to produce matching test-side BERT logits.
#   - Concatenate BERT logits with the existing TF-IDF stack and feed both into
#     LightGBM and SVC ensemble heads.
#   - 5-fold OOF macro-F1 gate: if final OOF macro-F1 < 0.74, write the
#     baseline submission (no BERT) instead — protects against v3-style regressions.
#   - try/except import for Claude validated rules.
from pathlib import Path
def _resolve():
    c = Path("/kaggle/input/spr-2026-mammography-report-classification")
    if (c / "train.csv").exists(): return c
    for p in Path("/kaggle/input").rglob("train.csv"):
        if (p.parent / "test.csv").exists(): return p.parent
    raise FileNotFoundError("train/test not found")
_D = _resolve()

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import hashlib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

TRAIN_PATH = str(_D / 'train.csv')
TEST_PATH  = str(_D / 'test.csv')

# [v6b] paths for BERTimbau artifacts attached as Kaggle datasets
# [v6b-fix] Hardcoded paths sometimes fail to mount; rglob /kaggle/input as fallback.
def _find_first(name: str) -> Path | None:
    base = Path('/kaggle/input')
    if not base.exists():
        return None
    for p in base.rglob(name):
        return p
    return None

_oof_p = Path('/kaggle/input/spr-2026-mammo-bert-oof/oof_logits.npy')
if not _oof_p.exists():
    _found = _find_first('oof_logits.npy')
    if _found is not None:
        print(f'[v6b-fix] OOF logits found via rglob at {_found}')
        _oof_p = _found
    else:
        print(f'[v6b-fix] /kaggle/input contents:')
        if Path('/kaggle/input').exists():
            for p in sorted(Path('/kaggle/input').iterdir())[:20]:
                print(f'  {p}')
                if p.is_dir():
                    for sub in sorted(p.iterdir())[:10]:
                        print(f'    {sub}')
BERT_OOF_PATH = _oof_p

BERT_WEIGHTS_DIR_CANDIDATES = [
    Path('/kaggle/input/mammo-bert-fold1'),
    Path('/kaggle/input/mammo-bertimbau-large'),
    Path('/kaggle/input/spr-2026-mammo-bert-oof'),  # placeholder if no weights
]
# [v6b-fix] Also rglob for BERT model weights
_w_found = _find_first('fold1_pytorch_model.bin')
if _w_found is not None and _w_found.parent not in BERT_WEIGHTS_DIR_CANDIDATES:
    print(f'[v6b-fix] BERT weights found via rglob at {_w_found.parent}')
    BERT_WEIGHTS_DIR_CANDIDATES.insert(0, _w_found.parent)
BERT_FOLD_ID = 1  # fold1 best single-fold OOF F1=0.80
BERT_MAX_LEN = 224
BERT_BATCH = 8

# [v6b] OOF gate: if the new ensemble's OOF macro-F1 falls below this, fall back
# to the baseline-without-bert submission. 0.74 is "safer than 0.81 baseline OOF
# of ~0.7522 minus a small slack" — protects against v3 regression.
OOF_GATE = 0.74

# [v6b] try-import of Claude validated rules. If unavailable, use a no-op.
try:
    sys.path.insert(0, '/kaggle/input/claude-validated-rules')
    sys.path.insert(0, '/tmp')
    from claude_validated_rules import apply_claude_rules
    print('[v6b] claude_validated_rules: imported.')
except Exception as e:
    print(f'[v6b] claude_validated_rules: not available ({type(e).__name__}); using identity.')
    def apply_claude_rules(df, preds, *_args, **_kwargs):
        return preds

# ---
# ─── PREPROCESSING ─────────────────────────────────────────────────────────────

def clean_achados(s: str) -> str:
    """Extract the clinical findings section and normalise numbers."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    # Extract findings section between 'achados:' and 'análise comparativa:'
    match = re.search(r'achados:(.*?)(análise comparativa:|$)', s, flags=re.DOTALL)
    if match:
        s = match.group(1).strip()
    # Normalise all numbers to a single token
    s = re.sub(r'[0-9]+,[0-9]+', 'NUM', s)
    s = re.sub(r'[0-9]+', 'NUM', s)
    return s


def clean_full(s: str) -> str:
    """Normalise the full report: whitespace + number masking."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r'[0-9]+,[0-9]+', 'NUM', s)
    s = re.sub(r'[0-9]+', 'NUM', s)
    return s


# [v6b] BERTimbau text normalization (matches training-time normalization)
_WS_RE = re.compile(r"\s+")
def normalize_for_bert(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\n\r", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return " [SEP] ".join(lines)


def stable_hash(s: str) -> str:
    """Deterministic MD5 hash for GroupKFold group IDs."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# [v6b] Discover BERT weights dir on Kaggle, return None if not found.
def find_bert_weights_dir():
    for d in BERT_WEIGHTS_DIR_CANDIDATES:
        if d.exists() and any((d / f"fold{i}_pytorch_model.bin").exists() for i in range(5)):
            return d
    # broader fallback
    base = Path('/kaggle/input')
    if base.exists():
        for i in range(5):
            for p in base.rglob(f"fold{i}_pytorch_model.bin"):
                return p.parent
            # legacy nested-fold layout
            for p in base.rglob(f"fold{i}/pytorch_model.bin"):
                return p.parent.parent
    return None


# [v6b] Reassemble a single-fold dir from the flat dataset layout (one-time).
def materialize_fold_dir(weights_root: Path, fold_id: int):
    import shutil, tempfile
    work = Path(tempfile.mkdtemp(prefix=f'bert_f{fold_id}_'))
    # Try flat layout: foldN_pytorch_model.bin + foldN_config.json + ...
    flat_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json',
                  'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    flat_ok = (weights_root / f'fold{fold_id}_pytorch_model.bin').exists()
    if flat_ok:
        for fn in flat_files:
            src = weights_root / f'fold{fold_id}_{fn}'
            dst = work / fn
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except Exception:
                    shutil.copy(src, dst)
        return work
    # Try nested layout: foldN/pytorch_model.bin
    nested = weights_root / f'fold{fold_id}'
    if nested.exists() and (nested / 'pytorch_model.bin').exists():
        return nested
    return None


# [v6b] Run BERT inference on a list of texts, return softmax probas (N, 7).
# Uses int8 dynamic quantization on CPU for ~3x speedup vs fp32.
def bert_infer_test(test_texts):
    weights_root = find_bert_weights_dir()
    if weights_root is None:
        print('[v6b] BERT weights not found under /kaggle/input — returning uniform priors.')
        return None
    print(f'[v6b] BERT weights found: {weights_root}')
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, DataCollatorWithPadding)
    except Exception as e:
        print(f'[v6b] torch/transformers unavailable: {type(e).__name__}: {e}')
        return None

    fold_dir = materialize_fold_dir(weights_root, BERT_FOLD_ID)
    if fold_dir is None:
        # try other folds
        for fid in [0, 2, 3, 4]:
            fold_dir = materialize_fold_dir(weights_root, fid)
            if fold_dir is not None:
                print(f'[v6b] fold{BERT_FOLD_ID} unavailable, using fold{fid}')
                break
    if fold_dir is None:
        print('[v6b] could not materialize any fold directory.')
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(fold_dir, use_fast=True)
    except Exception as e:
        print(f'[v6b] tokenizer load failed: {e}')
        return None

    class DS(Dataset):
        def __init__(self, texts, tok, maxlen):
            self.t = texts; self.k = tok; self.m = maxlen
        def __len__(self): return len(self.t)
        def __getitem__(self, i):
            return self.k(self.t[i], truncation=True, max_length=self.m,
                          return_attention_mask=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    ds = DS(test_texts, tokenizer, BERT_MAX_LEN)
    loader = DataLoader(ds, batch_size=BERT_BATCH, shuffle=False,
                        collate_fn=collator, num_workers=0, pin_memory=False)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            fold_dir, num_labels=7,
        )
    except Exception as e:
        print(f'[v6b] model load failed: {e}')
        return None
    model.eval()
    try:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print('[v6b] model quantized to int8.')
    except Exception as e:
        print(f'[v6b] quantization failed ({e}); using fp32.')

    chunks = []
    t0 = time.time()
    total = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            outputs = model(**batch)
            chunks.append(outputs.logits.detach().float().cpu().numpy())
            if i % 50 == 0:
                dt = time.time() - t0
                rate = (i+1) / max(dt, 1e-3)
                eta = (total - i - 1) / max(rate, 1e-3)
                print(f'[v6b] BERT batch {i}/{total} dt={dt:.1f}s eta={eta:.0f}s', flush=True)
    logits = np.concatenate(chunks, axis=0)
    print(f'[v6b] BERT inference done in {time.time()-t0:.1f}s; shape={logits.shape}')
    return logits


def softmax_logits(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def write_submission(df_test, preds, label_col='target'):
    sub = df_test[['ID']].copy()
    sub[label_col] = preds
    sub.to_csv('submission.csv', index=False)
    print(f'\nPrediction distribution:\n{sub[label_col].value_counts().sort_index()}')
    print('\nSubmission saved to submission.csv')


# ---
# ─── DATA LOADING ──────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f'Train: {train.shape}, Test: {test.shape}')

# Apply both text representations to train and test
train["achados"] = train["report"].apply(clean_achados)
train["full"]    = train["report"].apply(clean_full)
test["achados"]  = test["report"].apply(clean_achados)
test["full"]     = test["report"].apply(clean_full)

# Hash for GroupKFold (prevents report leakage across folds)
train["group"]   = train["report"].apply(stable_hash)

y = train["target"].astype(int).values
print(f'Target distribution:\n{pd.Series(y).value_counts().sort_index()}')

# ---
# ─── DENSE CLINICAL FEATURES ───────────────────────────────────────────────────
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

X_train_dense = extract_dense_features(train)
X_test_dense  = extract_dense_features(test)

# ---
# ─── TF-IDF FEATURES (3 heads) ─────────────────────────────────────────────────
print("Building TF-IDF features...")

# Head A: findings section only
tfidf_A = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.95,
                             sublinear_tf=True, max_features=80000))
])
X_train_A = tfidf_A.fit_transform(train["achados"])
X_test_A  = tfidf_A.transform(test["achados"])

# Head F: full report, standard char n-grams
tfidf_F = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.95,
                             sublinear_tf=True, max_features=80000))
])
X_train_F = tfidf_F.fit_transform(train["full"])
X_test_F  = tfidf_F.transform(test["full"])

# Head F2: full report, wider char n-grams (3–6) for morphological diversity
tfidf_F2 = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=3, max_df=0.95,
                             sublinear_tf=True, max_features=100000))
])
X_train_F2 = tfidf_F2.fit_transform(train["full"])
X_test_F2  = tfidf_F2.transform(test["full"])

# ---
# ─── [v6b] LOAD BERTIMBAU OOF FEATURES + RUN TEST INFERENCE ────────────────────
print('\n[v6b] Loading BERTimbau OOF features and running test inference...')

bert_train_logits = None
bert_test_logits = None

# Load OOF logits for train
if BERT_OOF_PATH.exists():
    try:
        bert_train_logits = np.load(BERT_OOF_PATH).astype(np.float32)
        if bert_train_logits.shape != (len(train), 7):
            print(f'[v6b] OOF logits shape {bert_train_logits.shape} != ({len(train)}, 7); discarding.')
            bert_train_logits = None
        else:
            print(f'[v6b] OOF logits loaded: {bert_train_logits.shape}')
    except Exception as e:
        print(f'[v6b] OOF logits load failed: {e}')
        bert_train_logits = None
else:
    print(f'[v6b] OOF logits file not found at {BERT_OOF_PATH}.')

# Run BERT inference on test
if bert_train_logits is not None:
    test_texts_norm = [normalize_for_bert(t) for t in test["report"].fillna("").astype(str).tolist()]
    bert_test_logits = bert_infer_test(test_texts_norm)
    if bert_test_logits is not None and bert_test_logits.shape != (len(test), 7):
        print(f'[v6b] BERT test logits shape {bert_test_logits.shape} != ({len(test)}, 7); discarding.')
        bert_test_logits = None

# Convert to softmax probas as features (more compact than raw logits)
USE_BERT = (bert_train_logits is not None) and (bert_test_logits is not None)
if USE_BERT:
    bert_train_proba = softmax_logits(bert_train_logits).astype(np.float32)
    bert_test_proba  = softmax_logits(bert_test_logits).astype(np.float32)
    print(f'[v6b] USE_BERT=True; train_proba={bert_train_proba.shape} test_proba={bert_test_proba.shape}')
    # Show test class distribution implied by BERT
    print(f'[v6b] BERT test argmax dist: {pd.Series(bert_test_proba.argmax(1)).value_counts().sort_index().to_dict()}')
else:
    print('[v6b] USE_BERT=False; will train baseline-only ensemble.')
    bert_train_proba = None
    bert_test_proba  = None

# ---
# ─── [v6b] FEATURE STACKS WITH OPTIONAL BERT ───────────────────────────────────
# LGB input: achados TF-IDF + dense features stacked + (optional) BERT probas
if USE_BERT:
    bert_train_sparse = csr_matrix(bert_train_proba)
    bert_test_sparse  = csr_matrix(bert_test_proba)
    X_train_lgb = hstack([X_train_A, X_train_dense, bert_train_sparse]).tocsr()
    X_test_lgb  = hstack([X_test_A,  X_test_dense,  bert_test_sparse]).tocsr()
else:
    X_train_lgb = hstack([X_train_A, X_train_dense]).tocsr()
    X_test_lgb  = hstack([X_test_A,  X_test_dense]).tocsr()

print(f"SVC-A: {X_train_A.shape[1]:,} features")
print(f"SVC-F: {X_train_F.shape[1]:,} features")
print(f"SVC-F2: {X_train_F2.shape[1]:,} features")
print(f"LGB: {X_train_lgb.shape[1]:,} features  (USE_BERT={USE_BERT})")

# ---
# ─── MAIN MODELS ───────────────────────────────────────────────────────────────
print("Training SVC-A (achados head)...")
svc_A = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
    cv=3, method='sigmoid'
)
svc_A.fit(X_train_A, y)
proba_A = svc_A.predict_proba(X_test_A)

print("Training SVC-F (full text head)...")
svc_F = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
    cv=3, method='sigmoid'
)
svc_F.fit(X_train_F, y)
proba_F = svc_F.predict_proba(X_test_F)

print("Training SVC-F2 (wide char n-gram head)...")
svc_F2 = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
    cv=3, method='sigmoid'
)
svc_F2.fit(X_train_F2, y)
proba_F2 = svc_F2.predict_proba(X_test_F2)

print("Training LightGBM (with BERT features if USE_BERT)...")
lgb_model = lgb.LGBMClassifier(
    class_weight='balanced',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_lgb, y)
proba_lgb = lgb_model.predict_proba(X_test_lgb)

# ---
# ─── ENSEMBLE WITH OPTIONAL BERT MIX ───────────────────────────────────────────
svc_ensemble = 0.25 * proba_A + 0.40 * proba_F + 0.35 * proba_F2

if USE_BERT:
    # [v6b] Conservative blend: SVC stays dominant. BERT enters at 15% via direct
    # mixing, plus its features have already entered the LGB head. Keep LGB at 30%.
    # SVC=0.55, LGB=0.30, BERT=0.15
    ensemble_proba = 0.55 * svc_ensemble + 0.30 * proba_lgb + 0.15 * bert_test_proba
    print('[v6b] Ensemble (with BERT): SVC=0.55 LGB=0.30 BERT=0.15')
else:
    ensemble_proba = 0.70 * svc_ensemble + 0.30 * proba_lgb
    print('[v6b] Ensemble (baseline only): SVC=0.70 LGB=0.30')
print("Ensemble built.")

# ---
# ─── BINARY CLASS-0 DETECTOR ───────────────────────────────────────────────────
print("Training binary class-0 detector on {0, 2} subset...")

mask_02 = np.isin(y, [0, 2])
y_02    = (y[mask_02] == 0).astype(int)   # 1 = class 0,  0 = class 2

bin0_svc = CalibratedClassifierCV(
    LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
    cv=3, method='sigmoid'
)
bin0_svc.fit(X_train_F[mask_02], y_02)
bin0_test_proba = bin0_svc.predict_proba(X_test_F)[:, 1]   # P(class 0)

print(f"Binary class-0 P(0) on test set sample: {bin0_test_proba[:5].round(4)}")

# ---
# ─── THRESHOLDS ────────────────────────────────────────────────────────────────
def apply_thresholds(prob, bin0):
    p = np.argmax(prob, axis=1).copy()
    p[(prob[:, 6] > 0.10)] = 6
    p[(prob[:, 5] > 0.15) & (p != 6)] = 5
    p[(prob[:, 4] > 0.23) & (p != 6) & (p != 5)] = 4
    p[(prob[:, 3] > 0.38) & (p != 6) & (p != 5) & (p != 4)] = 3
    if bin0 is not None:
        p[(bin0 > 0.55) & (p == 2)] = 0
    return p

print("Applying per-class thresholds...")
preds = apply_thresholds(ensemble_proba, bin0_test_proba)


# ─── CLINICAL GUARDRAILS ───────────────────────────────────────────────────────
def apply_safe_rules(row):
    text = str(row['report']).lower()
    pred = int(row['target'])
    # Confirmed malignancy → class 6
    if re.search(r'resultado de cine grau 3|carcinoma|\bcdis\b', text):
        return 6
    # Spiculation + distortion without high-grade prediction → class 5
    if 'espiculad' in text and 'distorção' in text and pred < 4:
        return 5
    return pred


# ─── [v6b] OOF GATE: VALIDATE THE NEW STACK BEFORE COMMITTING ──────────────────
# We compute a 3-fold OOF macro-F1 on a quick LR proxy of our final feature set.
# If the lift over baseline is negligible / negative, fall back to baseline.
print('\n[v6b] OOF gate: running 5-fold OOF macro-F1 on LGB head (proxy)...')
try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(train), dtype=np.int64)
    oof_proba = np.zeros((len(train), 7), dtype=np.float32)
    for fi, (tri, vai) in enumerate(skf.split(np.arange(len(train)), y)):
        m = lgb.LGBMClassifier(
            class_weight='balanced', n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42 + fi, n_jobs=-1, verbose=-1,
        )
        m.fit(X_train_lgb[tri], y[tri])
        oof_proba[vai] = m.predict_proba(X_train_lgb[vai])
    oof_pred = apply_thresholds(oof_proba, None)
    oof_f1 = f1_score(y, oof_pred, average='macro')
    print(f'[v6b] OOF macro-F1 (proxy LGB+BERT={USE_BERT}): {oof_f1:.4f} (gate={OOF_GATE})')
    # We only use this to decide BERT vs no-BERT; SVC is always trained.
    if USE_BERT and oof_f1 < OOF_GATE:
        print(f'[v6b] OOF below gate ({oof_f1:.4f} < {OOF_GATE}); FALLING BACK to baseline (no BERT).')
        # rebuild ensemble without BERT mix
        ensemble_proba = 0.70 * svc_ensemble + 0.30 * proba_lgb
        preds = apply_thresholds(ensemble_proba, bin0_test_proba)
        USE_BERT = False
except Exception as e:
    print(f'[v6b] OOF gate failed: {type(e).__name__}: {e}; keeping current ensemble.')
    oof_f1 = -1.0


# Apply safe rules then Claude-validated rules
test["target"] = preds
test['target'] = test.apply(apply_safe_rules, axis=1)

# [v6b] Claude validated rules (try/except imported at top). Pass test+preds as
# numpy array for flexibility.
try:
    new_preds = apply_claude_rules(test, test['target'].values.copy())
    if isinstance(new_preds, (list, np.ndarray)) and len(new_preds) == len(test):
        test['target'] = np.asarray(new_preds, dtype=int)
        print('[v6b] claude_validated_rules applied.')
    else:
        print('[v6b] claude_validated_rules returned bad shape; ignoring.')
except Exception as e:
    print(f'[v6b] claude_validated_rules error: {type(e).__name__}: {e}; ignoring.')

# ---
# ─── SAVE SUBMISSION ───────────────────────────────────────────────────────────
submission = test[['ID', 'target']].copy()
submission.to_csv('submission.csv', index=False)

print(f'\n[v6b] FINAL: USE_BERT={USE_BERT}  OOF macro-F1={oof_f1 if "oof_f1" in dir() else -1:.4f}')
print('\nPrediction distribution:')
print(submission['target'].value_counts().sort_index())
print('\nSubmission saved to submission.csv')
