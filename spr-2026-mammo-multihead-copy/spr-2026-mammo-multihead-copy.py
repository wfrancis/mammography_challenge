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
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

TRAIN_PATH = str(_D / 'train.csv')
TEST_PATH  = str(_D / 'test.csv')

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


def stable_hash(s: str) -> str:
    """Deterministic MD5 hash for GroupKFold group IDs."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()

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

# LGB input: achados TF-IDF + dense features stacked
X_train_lgb = hstack([X_train_A, X_train_dense]).tocsr()
X_test_lgb  = hstack([X_test_A,  X_test_dense]).tocsr()

print(f"SVC-A: {X_train_A.shape[1]:,} features")
print(f"SVC-F: {X_train_F.shape[1]:,} features")
print(f"SVC-F2: {X_train_F2.shape[1]:,} features")
print(f"LGB: {X_train_lgb.shape[1]:,} features")

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

print("Training LightGBM...")
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

# Weighted ensemble blend
svc_ensemble   = 0.25 * proba_A + 0.40 * proba_F + 0.35 * proba_F2
ensemble_proba = 0.70 * svc_ensemble + 0.30 * proba_lgb
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

print(f"Binary class-0 P(0) on test set: {bin0_test_proba.round(4)}")

# ---
# ─── THRESHOLDS ────────────────────────────────────────────────────────────────
print("Applying per-class thresholds...")

preds = np.argmax(ensemble_proba, axis=1).copy()

# High-severity class overrides (applied in priority order)
preds[(ensemble_proba[:, 6] > 0.15)] = 6
preds[(ensemble_proba[:, 5] > 0.20) & (preds != 6)] = 5
preds[(ensemble_proba[:, 4] > 0.25) & (preds != 6) & (preds != 5)] = 4
preds[(ensemble_proba[:, 3] > 0.35) & (preds != 6) & (preds != 5) & (preds != 4)] = 3

# Binary class-0 override: correct 0 vs 2 confusion
preds[(bin0_test_proba > 0.55) & (preds == 2)] = 0

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

test["target"] = preds
test['target'] = test.apply(apply_safe_rules, axis=1)

# ---
# ─── SAVE SUBMISSION ───────────────────────────────────────────────────────────
submission = test[['ID', 'target']].copy()
submission.to_csv('submission.csv', index=False)

print('\nPrediction distribution:')
print(submission['target'].value_counts().sort_index())
print('\nSubmission saved to submission.csv')