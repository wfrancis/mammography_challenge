"""SPR 2026 Mammography — PhD v3 Kernel.

v3 adds baseline CV + probability-space blending on top of v2. Stages 0-10
unchanged. New stages:

  Stage 11 Baseline 5-fold CV -> P_baseline_oof, bin0_oof, P_baseline_test.
  Stage 12 Probability blend: alpha * baseline + (1-alpha) * hierarchical,
           scanned over {0.20, 0.35, 0.50, 0.65, 0.80} on OOF.

Tier selector replaced. G1/G2/G5 gates removed; only G4_per_class_sane
remains as a quality floor. Candidates: Baseline, Hier, FlatRules, Blend.
v2_Tier4 baseline kept as emergency fallback if all candidates fail G4.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

N_CLASSES = 7          # labels 0..6
N_FOLDS = 5
JS_LAMBDA = 50.0
RULE_PREC_GATE = 0.95

# ---------------------------------------------------------------------------
# Environment detection: Kaggle vs. local
# ---------------------------------------------------------------------------

def resolve_paths() -> tuple[str, str]:
    """Return (train_path, test_path), preferring Kaggle mount, falling back
    to local data directory."""
    kaggle = Path("/kaggle/input/spr-2026-mammography-report-classification")
    if (kaggle / "train.csv").exists():
        return str(kaggle / "train.csv"), str(kaggle / "test.csv")
    for p in Path("/kaggle/input").rglob("train.csv") if Path("/kaggle/input").exists() else []:
        if (p.parent / "test.csv").exists():
            return str(p), str(p.parent / "test.csv")
    local = Path("./data/raw")
    if (local / "train.csv").exists():
        return str(local / "train.csv"), str(local / "test.csv")
    raise FileNotFoundError("train/test CSVs not found")


# ---------------------------------------------------------------------------
# Stage 0 — normalization, hashing, section splitting, GroupKFold
# ---------------------------------------------------------------------------

def norm(t: str) -> str:
    """Lowercase + collapse whitespace. Diacritics retained for TF-IDF."""
    t = str(t).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def ascii_fold(t: str) -> str:
    """Strip combining marks for regex features."""
    return "".join(c for c in unicodedata.normalize("NFKD", t)
                   if not unicodedata.combining(c))


def stable_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


_SEC_RE = re.compile(
    r"(indicac[ãa]o|achados|an[aá]lise\s+comparativa|"
    r"impress[ãa]o|conclus[ãa]o|bi-?rads)\s*[:\-]",
    re.IGNORECASE,
)


def split_sections(t: str) -> dict:
    """Tolerant section splitter. Returns a dict with empty strings for
    missing sections so downstream features never NaN."""
    spans = [(m.start(), m.group(1).lower()) for m in _SEC_RE.finditer(t)]
    spans.append((len(t), "END"))
    out = {"indicacao": "", "achados": "", "comparativa": "", "impressao": ""}
    for (s, name), (e, _) in zip(spans, spans[1:]):
        body = t[s:e]
        if name.startswith("ind"):
            out["indicacao"] += " " + body
        elif name.startswith("achados"):
            out["achados"] += " " + body
        elif name.startswith("an"):
            out["comparativa"] += " " + body
        elif name.startswith("impress") or name.startswith("conclus"):
            out["impressao"] += " " + body
    return out


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach text_raw/text_n/text_a and section columns, plus md5."""
    df = df.copy()
    # Spec uses `text`; actual column is `report`.
    df["text_raw"] = df["report"].fillna("")
    df["text_n"] = df["text_raw"].map(norm)
    df["text_a"] = df["text_n"].map(ascii_fold)
    df["md5"] = df["text_n"].map(stable_hash)
    secs = df["text_n"].map(split_sections)
    df["indicacao"] = secs.map(lambda s: s["indicacao"])
    df["achados"] = secs.map(lambda s: s["achados"])
    df["comparativa"] = secs.map(lambda s: s["comparativa"])
    df["impressao"] = secs.map(lambda s: s["impressao"])
    # ASCII-folded duplicates for section-scoped regex features
    df["achados_a"] = df["achados"].map(ascii_fold)
    df["impressao_a"] = df["impressao"].map(ascii_fold)
    return df


def assign_folds(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """GroupKFold(5) on md5. Returns an array of fold ids."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = np.full(len(df), -1, dtype=int)
    for i, (_, va) in enumerate(gkf.split(df, y, groups=df["md5"].values)):
        folds[va] = i
    assert (folds >= 0).all(), "GroupKFold missed a row"
    return folds


# ---------------------------------------------------------------------------
# Stage 1 — feature blocks (A/B/C tf-idf; H hand-crafted; S svd)
# ---------------------------------------------------------------------------

# Binary / count regex patterns.  Keys map directly to feature names.
_BIN_PATTERNS = {
    "has_carcinoma": r"\bcarcinoma\b",
    "has_invasivo": r"\binvasiv[oa]\b",
    "has_ductal": r"\bductal\b",
    "has_lobular": r"\blobular\b",
    "has_ca_confirm": r"\b(biopsi[ae]?|biopsia)\b.{0,40}\b(positiv|maligno|neoplasi)",
    "has_neoplasia": r"\bneoplasia\s+maligna\b",
    "has_resultado_anatomo": r"\banatom[oa]?\-?patolog",
    "has_espiculad": r"\bespiculad",
    "has_retracao": r"\bretra[cç]a[oõ]\b",
    "has_limites_imprecisos": r"\blimites?\s+imprecis",
    "has_contornos_irreg": r"\bcontornos?\s+irregulare?s?",
    "has_maior_que_antes": r"\baument(o|ou|ando)\b.{0,30}\b(nodulo|massa|lesao)",
    "has_densidade_alta": r"\balta\s+densidade\b",
    "has_pleomorfic": r"\bpleom[oó]rfica",
    "has_amorf": r"\bamorf",
    "has_agrupada": r"\bagrupad",
    "has_linear_ramificada": r"\blinear(es)?\s+ramificad",
    "has_extensao": r"\bextens[aã]o\b",
    "has_distribuicao_seg": r"\bdistribui[cç][aã]o\s+(segment|region)",
    "has_estavel": r"\best[aá]vel\b",
    "has_ha_anos": r"\bh[aá]\s+\d+\s+an[oa]s?\b",
    "has_assim_focal": r"\bassimetria\s+focal\b",
    "has_cisto_simples": r"\bcisto\s+simples\b",
    "has_fibroadenoma": r"\bfibroadenoma\b",
    "has_provavel_benign": r"\bprov[aá]vel\s+benign",
    "has_benignas_esparsas": r"\bbenignas?\s+esparsa",
    "has_calc_vascular": r"\bcalcifica[cç][aã]o\s+vascular",
    "has_calc_cutanea": r"\bcalcifica[cç][aã]o\s+cut[aâ]nea",
    "has_linfonodo_intram": r"\blinfonodo\s+intramamari",
    "has_hamart": r"\bhamartom",
    "has_negated_lesao": r"\bn[aã]o\s+(se\s+)?(observ|identific|evidenc|visualiz).{0,20}"
                        r"(nodul|massa|les[aã]o|calcifica)",
    "has_parcialmente_lipo": r"\bparcialmente\s+lipossubstitu",
    "has_tecido_fibro": r"\btecido\s+fibroglandular",
    "has_sem_achados": r"\bsem\s+(achados|alterac)",
    "has_nao_se_observam": r"\bn[aã]o\s+se\s+observam\b",
    "has_reavaliacao": r"\breavalia[cç][aã]o\b",
    "has_complementar": r"\bcomplementar\b",
    "has_ultrassom_adic": r"\bultrassom\s+(adicional|complementar)",
    "has_incidencia_adic": r"\bincid[eê]ncia\s+adicional",
    "has_muda_config": r"\bmuda\s+de\s+configura[cç][aã]o",
    "has_nao_caracterizad": r"\bn[aã]o\s+(totalmente\s+)?caracteriz",
    "br0": r"\bbi-?rads?\s*0\b",
    "br1": r"\bbi-?rads?\s*1\b",
    "br2": r"\bbi-?rads?\s*2\b",
    "br3": r"\bbi-?rads?\s*3\b",
    "br4": r"\bbi-?rads?\s*4[abc]?\b",
    "br5": r"\bbi-?rads?\s*5\b",
    "br6": r"\bbi-?rads?\s*6\b",
}

# v2 Addition A — expanded rare-class lexicon (merged into _BIN_PATTERNS).
_RARE_LEX_PATTERNS = {
    # Class 5 / 6 — malignancy markers
    "lex_retracao_cutanea":  r"retra[cç][ãa]o\s+cut[âa]nea",
    "lex_retracao_papila":   r"retra[cç][ãa]o\s+(?:da\s+)?pap[íi]la",
    "lex_nodulo_irregular":  r"n[óo]dulo\s+(?:de\s+contornos?\s+)?irregular",
    "lex_margem_espic":      r"margens?\s+espicul",
    "lex_margem_microlob":   r"margens?\s+microlobulada",
    "lex_contorno_irreg":    r"contornos?\s+irregulares",
    "lex_lesao_suspeita":    r"les[ãa]o\s+suspeita",
    "lex_caract_suspeita":   r"caracter[íi]sticas?\s+suspeita",
    "lex_altamente_sug":     r"altamente\s+sugestiv",
    "lex_sug_malignidade":   r"sugestiv.{0,20}maligni",
    "lex_achado_suspeito":   r"achado\s+suspeito",
    "lex_distorcao_arq":     r"distor[cç][ãa]o\s+arquitetural",
    "lex_hiperdens":         r"hiperdens",
    "lex_assim_focal":       r"assimetria\s+focal",
    "lex_assim_evolut":      r"assimetria\s+evolutiva",

    # Class 4 — calcifications suspicious
    "lex_calc_pleom_agrup":  r"calcifica[cç][õo]es\s+pleom[óo]rfica",
    "lex_calc_amorf_agrup":  r"calcifica[cç][õo]es\s+amorfas",
    "lex_calc_lin_ramif":    r"calcifica[cç][õo]es\s+lineares?\s+ramificada",
    "lex_calc_suspeitas":    r"calcifica[cç][õo]es\s+suspeita",
    "lex_dist_segmentar":    r"distribui[cç][ãa]o\s+segmentar",
    "lex_calc_heterogen":    r"calcifica[cç][õo]es\s+heterog[êe]nea",

    # Class 6 — confirmed cancer
    "lex_carcinoma_cdi":     r"\bcdi\b|carcinoma\s+ductal\s+invasiv",
    "lex_carcinoma_cdis":    r"\bcdis\b|carcinoma\s+ductal\s+in\s+situ",
    "lex_resultado_cine":    r"resultado\s+de\s+cine",
    "lex_core_biopsy_pos":   r"core\s+biopsy|core\s+bi[oó]psia",
    "lex_anatomo_malign":    r"anatomo?patolog.{0,40}(?:maligno|positiv|carcinoma)",
    "lex_neo_maligna":       r"neoplasia\s+maligna",
    "lex_invasivo":          r"\binvasiv[oa]\b",

    # Class 3 — probably benign
    "lex_prov_benign":       r"prov[áa]vel\s+benign",
    "lex_categoria_3":       r"categoria\s+3",
    "lex_controle_6m":       r"controle\s+em\s+6\s+m",
    "lex_nodulo_ovalado":    r"n[óo]dulo\s+oval",
    "lex_nodulo_circunscrito": r"n[óo]dulo\s+circunscri",

    # Class 0 — needs additional imaging
    "lex_incidencia_adic":   r"incid[êe]ncia\s+adicional",
    "lex_ultrassom_comp":    r"ultrassom\s+(?:adicional|complementar|dirigid)",
    "lex_comparacao_prev":   r"compara[cç][ãa]o\s+com\s+(?:exame\s+)?pr[ée]vi",
    "lex_necess_compl":      r"necess[áa]ri.{0,20}complement",

    # Conjunctions (powerful rare-class signals)
    "lex_nod_espic":         r"n[óo]dulo.{0,30}espiculad",
    "lex_nod_retrac":        r"n[óo]dulo.{0,80}retra[cç][ãa]o",
    "lex_espic_distor":      r"espiculad.{0,50}distor[cç][ãa]o",
    "lex_calc_pleom_agr2":   r"calcifica[cç][õo]es\s+pleom[óo]rficas\s+agrupadas",
    "lex_lesao_espic_med":   r"les[ãa]o.{0,30}espiculad.{0,50}\d+\s*(?:mm|cm)",
    "lex_linfono_axil_sus":  r"linfonodomegalia.{0,30}(?:suspeita|atipic)",

    # Negations / suppressors (protective for class 0/1/2)
    "lex_neg_malign":        r"n[ãa]o.{0,20}sinais?\s+de\s+malign",
    "lex_ausencia_nodulo":   r"aus[êe]ncia\s+de\s+n[óo]dulo",
    "lex_sem_alteracao":     r"sem\s+altera[cç][õo]es",
    "lex_categoria_1":       r"categoria\s+1",
    "lex_categoria_2":       r"categoria\s+2",
}

# Merge rare-lex into the main bin-pattern dict so `_BIN_RE` picks them up.
_BIN_PATTERNS.update(_RARE_LEX_PATTERNS)


_COUNT_PATTERNS = {
    "n_digits": r"\d",
    "n_measurements": r"\d+(?:[.,]\d+)?\s*(?:mm|cm)\b",
    "n_commas": r",",
    "n_negations": r"\bn[aã]o\b",
}

_SECTION_PATTERNS = [
    ("achados_has_carcinoma", "achados_a", r"\bcarcinoma\b"),
    ("achados_has_espiculad", "achados_a", r"\bespiculad"),
    ("achados_has_pleomorf", "achados_a", r"\bpleom[oó]rfic"),
    ("impressao_has_estavel", "impressao_a", r"\best[aá]vel\b"),
    ("impressao_has_benign", "impressao_a", r"\bbenign"),
]


def _compile(patterns):
    return {k: re.compile(p, flags=re.IGNORECASE) for k, p in patterns.items()}


_BIN_RE = _compile(_BIN_PATTERNS)
_COUNT_RE = _compile(_COUNT_PATTERNS)
_SECTION_RE = [(name, col, re.compile(p, re.IGNORECASE))
               for name, col, p in _SECTION_PATTERNS]


def handcrafted_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute ~80 hand-crafted dense features on ASCII-folded text.

    Returns (matrix[N, D], column_names).
    """
    N = len(df)
    text_a = df["text_a"].values
    achados_a = df["achados_a"].values
    impressao_a = df["impressao_a"].values
    text_n = df["text_n"].values

    cols = []
    mats = []

    # 1) binary regex features on text_a
    for name, rx in _BIN_RE.items():
        cols.append(name)
        mats.append(np.fromiter((1 if rx.search(t) else 0 for t in text_a),
                                dtype=np.float32, count=N))

    # 2) carcinoma scoped to achados (special dup)
    cols.append("has_carcinoma_achados")
    rx = re.compile(r"\bcarcinoma\b", re.IGNORECASE)
    mats.append(np.fromiter((1 if rx.search(a) else 0 for a in achados_a),
                            dtype=np.float32, count=N))

    # 3) count regex features
    for name, rx in _COUNT_RE.items():
        cols.append(name)
        mats.append(np.fromiter((len(rx.findall(t)) for t in text_a),
                                dtype=np.float32, count=N))

    # 4) section-scoped duplicates
    for name, col, rx in _SECTION_RE:
        cols.append(name)
        src = df[col].values
        mats.append(np.fromiter((1 if rx.search(s) else 0 for s in src),
                                dtype=np.float32, count=N))

    # 5) structural / length priors
    cols.append("len_chars")
    mats.append(np.array([len(t) for t in text_n], dtype=np.float32))
    cols.append("len_words")
    mats.append(np.array([len(t.split()) for t in text_n], dtype=np.float32))
    cols.append("len_achados")
    mats.append(np.array([len(a) for a in df["achados"].values], dtype=np.float32))
    cols.append("len_impressao")
    mats.append(np.array([len(a) for a in df["impressao"].values], dtype=np.float32))
    cols.append("frac_achados")
    mats.append(np.array([len(a) / max(1, len(t)) for a, t in
                          zip(df["achados"].values, text_n)], dtype=np.float32))
    cols.append("n_sections")
    mats.append(np.array([sum(1 for s in (df.iloc[i]["indicacao"],
                                          df.iloc[i]["achados"],
                                          df.iloc[i]["comparativa"],
                                          df.iloc[i]["impressao"])
                              if s.strip())
                          for i in range(N)], dtype=np.float32))

    X = np.vstack(mats).T.astype(np.float32)
    base_idx = {c: i for i, c in enumerate(cols)}

    # 6) interactions (computed from the base columns)
    interactions = {
        "carc_and_invasivo": lambda r: r[base_idx["has_carcinoma"]] * r[base_idx["has_invasivo"]],
        "espic_and_retrac":  lambda r: r[base_idx["has_espiculad"]] * r[base_idx["has_retracao"]],
        "pleom_and_agrup":   lambda r: r[base_idx["has_pleomorfic"]] * r[base_idx["has_agrupada"]],
        "benign_and_long":   lambda r: r[base_idx["has_benignas_esparsas"]] * (r[base_idx["len_chars"]] > 400),
    }
    extra_cols = list(interactions.keys())
    extra = np.zeros((N, len(extra_cols)), dtype=np.float32)
    for i in range(N):
        row = X[i]
        for j, name in enumerate(extra_cols):
            extra[i, j] = interactions[name](row)
    X = np.concatenate([X, extra], axis=1)
    cols.extend(extra_cols)
    return X, cols


# ---------------------------------------------------------------------------
# Stage 2 — base learners
# ---------------------------------------------------------------------------

def _align_proba(clf, proba: np.ndarray) -> np.ndarray:
    """Sklearn returns probabilities restricted to classes it saw during fit.
    We need a consistent [N, 7] array with label index == class label."""
    out = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float32)
    classes = np.asarray(clf.classes_, dtype=int)
    for j, cls in enumerate(classes):
        if 0 <= cls < N_CLASSES:
            out[:, cls] = proba[:, j]
    return out


def _fit_with_weights(clf, X_tr, y_tr, sample_weight=None):
    """Try to fit with sample_weight; fall back to no-weight on TypeError
    or estimators that don't accept it (e.g. CalibratedClassifierCV wraps
    LinearSVC and only passes kwargs it accepts)."""
    if sample_weight is not None:
        try:
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)
            return clf
        except TypeError:
            pass
        except ValueError:
            pass
    clf.fit(X_tr, y_tr)
    return clf


def _safe_fit_predict(clf_factory, X_tr, y_tr, X_va, sample_weight=None):
    """Fit a fresh classifier and return aligned predict_proba on X_va."""
    clf = clf_factory()
    clf = _fit_with_weights(clf, X_tr, y_tr, sample_weight=sample_weight)
    return clf, _align_proba(clf, clf.predict_proba(X_va))


def _fit_full(clf_factory, X, y, sample_weight=None):
    clf = clf_factory()
    clf = _fit_with_weights(clf, X, y, sample_weight=sample_weight)
    return clf


# Factory functions — called once per fold and once on the full set.

def make_M1():
    # Multinomial LR on [A;H_scaled].  Do not pass `multi_class` under new sklearn.
    return LogisticRegression(penalty="l2", C=4.0, solver="lbfgs",
                              class_weight="balanced", max_iter=2000,
                              n_jobs=-1, random_state=SEED)


def make_M2():
    return ComplementNB(alpha=0.3)


def make_M3():
    base = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000,
                     random_state=SEED)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def make_M4():
    return LogisticRegression(C=2.0, class_weight="balanced",
                              solver="lbfgs", max_iter=2000,
                              n_jobs=-1, random_state=SEED)


def make_M5():
    return HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.06, max_iter=400,
        l2_regularization=1.0, min_samples_leaf=20,
        random_state=SEED, early_stopping=True,
        validation_fraction=0.15)


# ---------------------------------------------------------------------------
# Stage 4 — per-class isotonic calibration (Platt fallback)
# ---------------------------------------------------------------------------

def calibrate_oof(q_oof: np.ndarray, y: np.ndarray, min_n: int = 50):
    """Fit per-class 1-vs-rest calibrators.  Returns a list[(kind, model)].

    Handles 3 cases:
      - yk.sum() == 0: class absent in training subset (e.g. Stage 2c non-class-2
        head). Store ('absent', None); apply_calibration returns 0 for that column.
      - 0 < yk.sum() < min_n: Platt fallback (LR on scalar input).
      - yk.sum() >= min_n: isotonic regression.
    """
    cals = []
    for k in range(N_CLASSES):
        yk = (y == k).astype(int)
        qk = q_oof[:, k]
        pos = int(yk.sum())
        if pos == 0:
            cals.append(("absent", None))
        elif pos < min_n:
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(qk.reshape(-1, 1), yk)
            cals.append(("platt", clf))
        else:
            iso = IsotonicRegression(out_of_bounds="clip").fit(qk, yk)
            cals.append(("iso", iso))
    return cals


def apply_calibration(q: np.ndarray, cals) -> np.ndarray:
    """Apply per-class calibrators to q and renormalise to simplex.

    'absent' classes (no positives in training) get set to 0 before renorm,
    so they can never win argmax in the hierarchical 6-way head.
    """
    out = np.zeros_like(q, dtype=np.float64)
    for k, (kind, model) in enumerate(cals):
        if kind == "iso":
            out[:, k] = model.predict(q[:, k])
        elif kind == "platt":
            out[:, k] = model.predict_proba(q[:, k].reshape(-1, 1))[:, 1]
        else:  # absent
            out[:, k] = 0.0
    out = np.clip(out, 1e-9, 1.0)
    out = out / out.sum(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Stage 5 — threshold tuning with coordinate ascent + James-Stein
# ---------------------------------------------------------------------------

def predict_with_tau(q: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.argmax(q / tau[None, :], axis=1)


def tune_tau(q_oof: np.ndarray, y: np.ndarray,
             n_iter: int = 30, lam_js: float = JS_LAMBDA,
             verbose: bool = False) -> tuple[np.ndarray, float, np.ndarray]:
    """Coordinate ascent over tau to maximise OOF macro-F1, with
    James-Stein shrinkage toward 1.0.  Returns (tau_js, best_f1_raw, tau_raw).

    `best_f1_raw` is the macro-F1 at the *unshrunk* tau.
    """
    grid = np.concatenate([np.linspace(0.05, 1.0, 20),
                           np.linspace(1.05, 3.0, 20)])
    K = q_oof.shape[1]
    tau = np.ones(K)
    n_k = np.bincount(y, minlength=K).astype(float)
    best = f1_score(y, predict_with_tau(q_oof, tau), average="macro")
    for it in range(n_iter):
        improved = False
        for k in np.random.default_rng(it).permutation(K):
            f_best = best
            t_best = tau[k]
            for t in grid:
                tau[k] = t
                f = f1_score(y, predict_with_tau(q_oof, tau), average="macro")
                if f > f_best + 1e-6:
                    f_best, t_best = f, t
            tau[k] = t_best
            if f_best > best + 1e-6:
                best = f_best
                improved = True
        if verbose:
            print(f"  tune_tau iter {it}: best={best:.4f}")
        if not improved:
            break
    tau_js = (n_k / (n_k + lam_js)) * tau + (lam_js / (n_k + lam_js)) * 1.0
    return tau_js, best, tau


# ---------------------------------------------------------------------------
# Bootstrap p05 (group-level)
# ---------------------------------------------------------------------------

def bootstrap_p05(q: np.ndarray, y: np.ndarray, tau: np.ndarray,
                  groups: np.ndarray, n_boot: int = 200,
                  seed: int = SEED) -> float:
    """Resample groups (not rows) and return 5th-percentile OOF macro-F1."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # Precompute row indices by group for O(1) lookup.
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    scores = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        pred = predict_with_tau(q[idx], tau)
        scores.append(f1_score(y[idx], pred, average="macro"))
    return float(np.percentile(scores, 5))


# ---------------------------------------------------------------------------
# v2 Addition G — inverse-frequency^power sample weights
# ---------------------------------------------------------------------------

def inv_freq_weights(y: np.ndarray, power: float = 0.7) -> np.ndarray:
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1
    w_class = 1.0 / (counts ** power)
    w_class = w_class / w_class.mean()
    return w_class[y].astype(np.float32)


# ---------------------------------------------------------------------------
# v2 Addition D — differential-evolution threshold search
# ---------------------------------------------------------------------------

def tune_tau_de(q_oof: np.ndarray, y: np.ndarray,
                lam_js: float = JS_LAMBDA, maxiter: int = 80,
                seed: int = SEED) -> tuple[np.ndarray, float, np.ndarray]:
    """Differential-evolution over tau. Returns (tau_js, best_f1_raw, tau_raw)."""
    try:
        from scipy.optimize import differential_evolution
    except Exception as e:
        raise RuntimeError(f"scipy DE unavailable: {e}")
    K = q_oof.shape[1]
    n_k = np.bincount(y, minlength=K).astype(float)

    def neg_f1(tau):
        tau_arr = np.asarray(tau, dtype=np.float64)
        pred = np.argmax(q_oof / tau_arr[None, :], axis=1)
        return -f1_score(y, pred, average="macro")

    res = differential_evolution(
        neg_f1, bounds=[(0.1, 4.0)] * K, maxiter=maxiter, seed=seed,
        polish=True, tol=1e-5, popsize=15, workers=1,
    )
    tau_raw = np.asarray(res.x, dtype=np.float64)
    tau_js = (n_k / (n_k + lam_js)) * tau_raw + (lam_js / (n_k + lam_js)) * 1.0
    return tau_js, float(-res.fun), tau_raw


# ---------------------------------------------------------------------------
# v2 Addition C — hierarchical (binary + 6-way) merge
# ---------------------------------------------------------------------------

def hierarchical_predict(p_bin: np.ndarray, q_not2_cal: np.ndarray,
                         tau_bin: float, tau_not2: np.ndarray) -> np.ndarray:
    """p_bin: [N] P(y==2).  q_not2_cal: [N, 7].  Returns argmax class."""
    preds = np.zeros(len(p_bin), dtype=int)
    use_stage2 = p_bin <= tau_bin
    preds[~use_stage2] = 2
    tau_safe = np.asarray(tau_not2, dtype=np.float64).copy()
    tau_safe[tau_safe <= 0] = 1.0
    if use_stage2.any():
        sub = q_not2_cal[use_stage2] / tau_safe[None, :]
        preds[use_stage2] = np.argmax(sub, axis=1)
    return preds


def tune_hierarchical(p_bin_oof: np.ndarray, q_not2_oof_cal: np.ndarray,
                      y: np.ndarray,
                      bin_grid=None, n_iter: int = 30):
    """Joint tune of (tau_bin, tau_not2) on OOF macro-F1.
    tau_not2 uses coord-ascent (tune_tau) on the subset; DE is NOT used here
    because the subset is small.
    """
    if bin_grid is None:
        bin_grid = np.linspace(0.6, 0.98, 20)
    best_f1 = -1.0
    best_t1 = 0.5
    best_tau = np.ones(N_CLASSES)
    for t1 in bin_grid:
        mask = p_bin_oof <= t1
        if mask.sum() < 100:
            continue
        sub_q = q_not2_oof_cal[mask]
        sub_y = y[mask]
        tau, _f_sub, _ = tune_tau(sub_q, sub_y, n_iter=n_iter)
        preds = hierarchical_predict(p_bin_oof, q_not2_oof_cal, t1, tau)
        f1 = f1_score(y, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t1, best_tau = f1, float(t1), tau
    return best_t1, best_tau, best_f1


# ---------------------------------------------------------------------------
# v2 Addition E — pseudo-label round
# ---------------------------------------------------------------------------

def pseudo_label_round(q_test_cal: np.ndarray, threshold: float = 0.95):
    """Return (indices_into_test, pseudo_labels). Never pseudo-labels class 2."""
    topk = q_test_cal.argmax(axis=1)
    topp = q_test_cal.max(axis=1)
    mask = (topp >= threshold) & (topk != 2)
    return np.where(mask)[0], topk[mask].astype(int)


def pseudo_prior_ok(y_pl: np.ndarray, y_train: np.ndarray,
                    max_ratio: float = 2.0) -> tuple[bool, np.ndarray]:
    """Reject if any class in pseudo-set deviates >max_ratio from train prior."""
    if len(y_pl) == 0:
        return False, np.zeros(N_CLASSES, dtype=int)
    train_prior = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    train_prior = train_prior / max(1, train_prior.sum())
    pl_dist = np.bincount(y_pl, minlength=N_CLASSES).astype(float)
    pl_counts = pl_dist.astype(int).copy()
    pl_dist = pl_dist / max(1, pl_dist.sum())
    for k in range(N_CLASSES):
        # class-2 excluded by construction
        if k == 2:
            continue
        tp = train_prior[k]
        pp = pl_dist[k]
        if tp <= 0 and pp > 0:
            # Pseudo-class absent in train but present in PL -> reject
            return False, pl_counts
        if tp > 0 and (pp > max_ratio * tp or (pp > 0 and tp > max_ratio * pp)):
            return False, pl_counts
    return True, pl_counts


# ---------------------------------------------------------------------------
# v2 Addition F — MD5 + Jaccard lookup layer
# ---------------------------------------------------------------------------

def _lookup_normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def build_lookup(train_df: pd.DataFrame):
    """Builds the lookup structure from train rows (report + target)."""
    from collections import Counter
    from sklearn.feature_extraction.text import CountVectorizer

    df = train_df.copy()
    df["report_n"] = df["report"].astype(str).map(_lookup_normalize)
    df["md5"] = df["report_n"].map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())

    md5_to_label = {}
    md5_to_conflict = {}
    for m, g in df.groupby("md5"):
        c = Counter(g["target"].astype(int).tolist())
        top, n = c.most_common(1)[0]
        md5_to_label[m] = int(top)
        md5_to_conflict[m] = (n / len(g) < 0.80)

    vec = CountVectorizer(analyzer="word", ngram_range=(3, 3),
                          min_df=2, max_features=300_000,
                          binary=True, dtype=np.int8)
    Xtr = vec.fit_transform(df["report_n"].str.lower())
    row_sums_tr = np.asarray(Xtr.sum(axis=1)).ravel().astype(np.float32)
    labels_tr = df["target"].astype(int).values

    return dict(
        vec=vec, Xtr=Xtr, row_sums_tr=row_sums_tr,
        labels_tr=labels_tr, md5_to_label=md5_to_label,
        md5_to_conflict=md5_to_conflict, md5_arr=df["md5"].values,
    )


def lookup_predict(test_df: pd.DataFrame, lk: dict,
                   tau_jaccard: float = 0.97, chunk: int = 500):
    """Returns (preds[-1 for miss], source[...], best_sim[...])."""
    rep = test_df["report"].astype(str).map(_lookup_normalize)
    hashes = rep.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    n = len(test_df)
    preds = np.full(n, -1, dtype=np.int32)
    source = np.array(["miss"] * n, dtype=object)
    best_sim = np.zeros(n, dtype=np.float32)

    for i, h in enumerate(hashes):
        if h in lk["md5_to_label"]:
            preds[i] = lk["md5_to_label"][h]
            source[i] = "exact_conflict" if lk["md5_to_conflict"][h] else "exact"

    todo = np.where(preds == -1)[0]
    if len(todo):
        Xte = lk["vec"].transform(rep.iloc[todo].str.lower())
        rs_te = np.asarray(Xte.sum(axis=1)).ravel().astype(np.float32)
        Xtr, rs_tr, labs = lk["Xtr"], lk["row_sums_tr"], lk["labels_tr"]
        for s in range(0, Xte.shape[0], chunk):
            e = min(s + chunk, Xte.shape[0])
            inter = (Xte[s:e] @ Xtr.T).toarray().astype(np.float32)
            union = rs_te[s:e, None] + rs_tr[None, :] - inter
            union[union == 0] = 1
            jac = inter / union
            j = jac.argmax(axis=1)
            v = jac[np.arange(e - s), j]
            for k_i, (jk, vk) in enumerate(zip(j, v)):
                idx = todo[s + k_i]
                best_sim[idx] = float(vk)
                if vk >= tau_jaccard:
                    preds[idx] = int(labs[jk])
                    source[idx] = "fuzzy"
    return preds, source, best_sim


def apply_lookup_tiebreak(test_df: pd.DataFrame, preds: np.ndarray,
                          q_test_cal: np.ndarray, train_df: pd.DataFrame,
                          tau_j: float = 0.97) -> tuple[np.ndarray, int]:
    """Lookup fires ONLY when pred in {1,2} and top prob < 0.85. Never overrides
    rare-class predictions (4/5/6). Returns (new_preds, n_flips)."""
    lk = build_lookup(train_df)
    lk_preds, lk_src, _ = lookup_predict(test_df, lk, tau_jaccard=tau_j)
    out = preds.copy()
    topp = q_test_cal.max(axis=1)
    flips = 0
    for i, p in enumerate(preds):
        if lk_preds[i] == -1 or lk_src[i] == "exact_conflict":
            continue
        # Only fire for predicted class in {1,2}; never override rare classes.
        if p in (1, 2) and lk_preds[i] in (1, 2) and topp[i] < 0.85:
            if int(lk_preds[i]) != int(out[i]):
                out[i] = int(lk_preds[i])
                flips += 1
    return out, flips


def apply_lookup_tiebreak_oof(train_df: pd.DataFrame, preds_oof: np.ndarray,
                              q_oof_cal: np.ndarray,
                              tau_j: float = 0.97) -> tuple[np.ndarray, int]:
    """Leave-one-MD5-group-out simulation of the lookup tie-break over OOF.
    Each row's lookup uses a train frame with rows sharing its MD5 removed."""
    from collections import Counter
    from sklearn.feature_extraction.text import CountVectorizer

    df = train_df.copy()
    df["report_n"] = df["report"].astype(str).map(_lookup_normalize)
    df["md5"] = df["report_n"].map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    md5_arr = df["md5"].values
    labels = df["target"].astype(int).values

    # Per-md5 label counts -> leave-one-group-out exact lookup per row.
    md5_to_counter = {}
    for m, g in df.groupby("md5"):
        md5_to_counter[m] = Counter(g["target"].astype(int).tolist())

    # Fuzzy vectorization for LOO is expensive; skip fuzzy in the OOF simulation
    # and only use exact-match LOO majority. This is a conservative proxy:
    # flips by fuzzy on OOF would be rare (rare classes have no fuzzy twins).
    vec = CountVectorizer(analyzer="word", ngram_range=(3, 3),
                          min_df=2, max_features=300_000,
                          binary=True, dtype=np.int8)
    try:
        Xtr = vec.fit_transform(df["report_n"].str.lower())
    except ValueError:
        Xtr = None

    out = preds_oof.copy()
    topp = q_oof_cal.max(axis=1)
    flips = 0
    for i, p in enumerate(preds_oof):
        if p not in (1, 2) or topp[i] >= 0.85:
            continue
        m = md5_arr[i]
        cnt = md5_to_counter.get(m)
        if cnt is None:
            continue
        # Remove this row's contribution.
        y_i = int(labels[i])
        cnt_loo = cnt.copy()
        cnt_loo[y_i] -= 1
        if cnt_loo[y_i] <= 0:
            del cnt_loo[y_i]
        if not cnt_loo:
            continue
        top_label, top_n = cnt_loo.most_common(1)[0]
        total = sum(cnt_loo.values())
        conflict = (top_n / total < 0.80) if total > 0 else True
        if conflict:
            continue
        if top_label in (1, 2) and int(top_label) != int(out[i]):
            out[i] = int(top_label)
            flips += 1
    return out, flips


# ---------------------------------------------------------------------------
# Stage 6 — precision-gated rule overrides
# ---------------------------------------------------------------------------

_R_ACHADOS_CARC = re.compile(r"\bcarcinoma\b", re.IGNORECASE)
_R_CA_CONFIRM = re.compile(r"\b(biopsi[ae]?|anatomo?patolog)\b.{0,60}"
                           r"\b(maligno|positiv|neoplasi|carcinoma)",
                           re.IGNORECASE | re.S)
_R_ESPIC_RETRAC = re.compile(r"\bespiculad.{0,80}retra[cç]a[oõ]",
                             re.IGNORECASE | re.S)
_R_BENIGN_ESPARS = re.compile(r"\bbenignas?\s+esparsa", re.IGNORECASE)
_R_CA_NEG = re.compile(r"\b(aus[eê]ncia|sem\s+evid[eê]ncia|"
                       r"n[aã]o\s+h[aá]|descartad).{0,30}"
                       r"(carcinoma|neoplasi|malign)",
                       re.IGNORECASE | re.S)


def compute_rule_masks(text_a: np.ndarray, achados_a: np.ndarray):
    """Precompute boolean masks for each rule's trigger."""
    N = len(text_a)
    m_neg = np.array([bool(_R_CA_NEG.search(t)) for t in text_a])
    m_r6 = np.array([(bool(_R_ACHADOS_CARC.search(a))
                      or bool(_R_CA_CONFIRM.search(t)))
                     for t, a in zip(text_a, achados_a)])
    m_r5 = np.array([(bool(_R_ESPIC_RETRAC.search(t))
                      and not bool(_R_ACHADOS_CARC.search(a)))
                     for t, a in zip(text_a, achados_a)])
    m_r12 = np.array([bool(_R_BENIGN_ESPARS.search(t)) and len(t) > 400
                      for t in text_a])
    return {
        "R_neg": m_neg,
        "R6": m_r6,
        "R5": m_r5,
        "R1to2": m_r12,
    }


def rule_oof_precisions(masks: dict, pred_oof: np.ndarray,
                        y: np.ndarray) -> dict:
    """Measure each rule's precision on OOF predictions."""
    neg = masks["R_neg"]
    out = {}

    # R6: precision over rows where the rule fires (excluding R_neg suppression).
    denom6 = masks["R6"] & ~neg
    if denom6.any():
        out["R6"] = float((y[denom6] == 6).mean())
    else:
        out["R6"] = 0.0

    # R5: espic+retrac, no carcinoma in achados, not suppressed.
    denom5 = masks["R5"] & ~masks["R6"] & ~neg
    if denom5.any():
        out["R5"] = float((y[denom5] == 5).mean())
    else:
        out["R5"] = 0.0

    # R1to2: stack predicted 1 AND regex fires (+ length). Precision: y==2.
    denom12 = masks["R1to2"] & (pred_oof == 1) & ~masks["R6"] & ~masks["R5"] & ~neg
    if denom12.any():
        out["R1to2"] = float((y[denom12] == 2).mean())
    else:
        out["R1to2"] = 0.0
    return out


def apply_rules(pred: np.ndarray, masks: dict, precisions: dict,
                gate: float = RULE_PREC_GATE) -> np.ndarray:
    """Apply rule overrides in spec order, subject to precision gate."""
    y = pred.copy()
    neg = masks["R_neg"]
    pass6 = precisions.get("R6", 0.0) >= gate
    pass5 = precisions.get("R5", 0.0) >= gate
    pass12 = precisions.get("R1to2", 0.0) >= gate
    for i in range(len(y)):
        if neg[i]:
            continue
        if pass6 and masks["R6"][i]:
            y[i] = 6
            continue
        if pass5 and masks["R5"][i]:
            y[i] = 5
            continue
        if pass12 and y[i] == 1 and masks["R1to2"][i]:
            y[i] = 2
            continue
    return y


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    train_path, test_path = resolve_paths()
    print(f"[paths] train={train_path}")
    print(f"[paths] test ={test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"[data] train={train.shape}  test={test.shape}")

    train = build_frame(train)
    test = build_frame(test)
    y = train["target"].astype(int).values
    print(f"[data] label counts: {np.bincount(y, minlength=N_CLASSES).tolist()}")

    # ----------------------------------------------------------------- Stage 0
    folds = assign_folds(train, y)
    print(f"[stage0] fold sizes: "
          f"{[int((folds == k).sum()) for k in range(N_FOLDS)]}")

    # ----------------------------------------------------------------- Stage 1
    print("[stage1] fitting TF-IDF blocks...")
    corpus_full = pd.concat([train["text_n"], test["text_n"]], ignore_index=True)
    corpus_achados = pd.concat([train["achados"], test["achados"]], ignore_index=True)

    A = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                        max_df=0.95, sublinear_tf=True, max_features=40000,
                        strip_accents=None, lowercase=False)
    A.fit(corpus_full)
    A_tr = A.transform(train["text_n"]).astype(np.float32)
    A_te = A.transform(test["text_n"]).astype(np.float32)

    B = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                        sublinear_tf=True, max_features=40000)
    B.fit(corpus_full)
    B_tr = B.transform(train["text_n"]).astype(np.float32)
    B_te = B.transform(test["text_n"]).astype(np.float32)

    C = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                        sublinear_tf=True, max_features=10000)
    C.fit(corpus_achados)
    C_tr = C.transform(train["achados"]).astype(np.float32)
    C_te = C.transform(test["achados"]).astype(np.float32)

    print(f"[stage1] A={A_tr.shape[1]} B={B_tr.shape[1]} C={C_tr.shape[1]}")

    # Hand-crafted dense features
    H_tr, H_cols = handcrafted_features(train)
    H_te, _ = handcrafted_features(test)
    print(f"[stage1] H features: {H_tr.shape[1]}")

    # Scaled H for linear learners; raw H for GBM.
    H_scaler = StandardScaler(with_mean=False).fit(H_tr)
    H_tr_sc = H_scaler.transform(H_tr).astype(np.float32)
    H_te_sc = H_scaler.transform(H_te).astype(np.float32)

    # TruncatedSVD(256) over [A;B]
    print("[stage1] fitting SVD-256...")
    svd_input_tr = hstack([A_tr, B_tr]).tocsr()
    svd_input_te = hstack([A_te, B_te]).tocsr()
    svd = TruncatedSVD(n_components=256, random_state=SEED)
    svd.fit(svd_input_tr)  # SVD fit on train only (spec uses train+test; train is fine & cheaper)
    S_tr = svd.transform(svd_input_tr).astype(np.float32)
    S_te = svd.transform(svd_input_te).astype(np.float32)

    # M1 input = [A ; H_sc]
    M1_tr_full = hstack([A_tr, csr_matrix(H_tr_sc)]).tocsr()
    M1_te_full = hstack([A_te, csr_matrix(H_te_sc)]).tocsr()

    # M2 input = A (non-negative)
    M2_tr_full = A_tr
    M2_te_full = A_te

    # M3 input = B
    M3_tr_full = B_tr
    M3_te_full = B_te

    # M4 input = S (dense)
    M4_tr_full = S_tr
    M4_te_full = S_te

    # M5 input = [H ; chi2_top400(A) ; chi2_top200(C)] dense
    # Chi2 selectors are *fit on training only* to avoid label leakage.
    print("[stage1] fitting chi2 selectors for M5...")
    sel_A = SelectKBest(chi2, k=min(400, A_tr.shape[1])).fit(A_tr, y)
    sel_C = SelectKBest(chi2, k=min(200, C_tr.shape[1])).fit(C_tr, y)
    A_top_tr = sel_A.transform(A_tr).toarray().astype(np.float32)
    A_top_te = sel_A.transform(A_te).toarray().astype(np.float32)
    C_top_tr = sel_C.transform(C_tr).toarray().astype(np.float32)
    C_top_te = sel_C.transform(C_te).toarray().astype(np.float32)
    M5_tr_full = np.concatenate([H_tr, A_top_tr, C_top_tr], axis=1).astype(np.float32)
    M5_te_full = np.concatenate([H_te, A_top_te, C_top_te], axis=1).astype(np.float32)
    print(f"[stage1] M5 dense shape: {M5_tr_full.shape}")

    # ----------------------------------------------------------------- Stage 2
    print("[stage2] training base learners on GroupKFold(5)...")
    N = len(train)
    P_oof = np.zeros((5, N, N_CLASSES), dtype=np.float32)
    base_factories = [make_M1, make_M2, make_M3, make_M4, make_M5]
    base_inputs_tr = [M1_tr_full, M2_tr_full, M3_tr_full, M4_tr_full, M5_tr_full]
    base_inputs_te = [M1_te_full, M2_te_full, M3_te_full, M4_te_full, M5_te_full]
    base_names = ["M1_LR", "M2_CNB", "M3_SVC", "M4_LR_SVD", "M5_HGB"]

    # v2 Addition G — inverse-frequency^0.7 sample weights (alongside
    # `class_weight='balanced'` already used by the base factories).
    w_full = inv_freq_weights(y, power=0.7)

    for f in range(N_FOLDS):
        tr_mask = folds != f
        va_mask = folds == f
        print(f"  fold {f}: train={tr_mask.sum()}, val={va_mask.sum()}")
        for i, (fac, Xtr_full) in enumerate(zip(base_factories, base_inputs_tr)):
            Xtr = Xtr_full[tr_mask]
            Xva = Xtr_full[va_mask]
            ytr = y[tr_mask]
            ts = time.time()
            sw = w_full[tr_mask]
            clf, proba = _safe_fit_predict(lambda f=fac: f(), Xtr, ytr, Xva,
                                           sample_weight=sw)
            P_oof[i, va_mask] = proba
            print(f"    {base_names[i]}: {time.time()-ts:.1f}s")

    # Per-base OOF macro-F1 (diagnostic only).
    base_f1 = []
    for i in range(5):
        f1 = f1_score(y, P_oof[i].argmax(axis=1), average="macro")
        base_f1.append(f1)
        print(f"[stage2] {base_names[i]} OOF macro-F1: {f1:.4f}")

    # Full-fit base learners for test-time predictions.
    print("[stage2] fitting base learners on full train for test predictions...")
    P_test = np.zeros((5, len(test), N_CLASSES), dtype=np.float32)
    for i, (fac, Xtr, Xte) in enumerate(zip(base_factories, base_inputs_tr,
                                            base_inputs_te)):
        ts = time.time()
        clf = _fit_full(lambda f=fac: f(), Xtr, y, sample_weight=w_full)
        P_test[i] = _align_proba(clf, clf.predict_proba(Xte))
        print(f"  {base_names[i]} full-fit: {time.time()-ts:.1f}s")

    # ---------------------------------------------------------------- Stage 2b
    # Binary (y==2) head across GroupKFold(5) on [A;H_scaled] = M1_tr_full.
    print("[stage2b] training binary (y==2) head on GroupKFold(5)...")
    y_bin = (y == 2).astype(int)
    P_bin_oof = np.zeros(N, dtype=np.float32)
    for f in range(N_FOLDS):
        tr_mask = folds != f
        va_mask = folds == f
        clf_bin_f = LogisticRegression(C=2.0, class_weight="balanced",
                                       solver="lbfgs", max_iter=2000,
                                       n_jobs=-1, random_state=SEED)
        try:
            clf_bin_f.fit(M1_tr_full[tr_mask], y_bin[tr_mask])
        except Exception as e:
            print(f"  [stage2b] fold {f} fit failed ({e}); default=prior")
            P_bin_oof[va_mask] = y_bin[tr_mask].mean()
            continue
        classes = list(clf_bin_f.classes_)
        if 1 in classes:
            idx_pos = classes.index(1)
            P_bin_oof[va_mask] = clf_bin_f.predict_proba(M1_tr_full[va_mask])[:, idx_pos]
        else:
            # Degenerate fold (no positives); use class-prior fallback.
            P_bin_oof[va_mask] = 0.0
    bin_ll = float(np.mean(
        -(y_bin * np.log(np.clip(P_bin_oof, 1e-7, 1 - 1e-7))
          + (1 - y_bin) * np.log(np.clip(1 - P_bin_oof, 1e-7, 1 - 1e-7)))
    ))
    print(f"[stage2b] binary head OOF logloss: {bin_ll:.4f}")

    clf_bin_full = LogisticRegression(C=2.0, class_weight="balanced",
                                      solver="lbfgs", max_iter=2000,
                                      n_jobs=-1, random_state=SEED)
    clf_bin_full.fit(M1_tr_full, y_bin)
    classes_full = list(clf_bin_full.classes_)
    if 1 in classes_full:
        P_bin_test = clf_bin_full.predict_proba(M1_te_full)[:, classes_full.index(1)]
    else:
        P_bin_test = np.zeros(len(test), dtype=np.float32)

    # ---------------------------------------------------------------- Stage 2c
    # 6-way stack trained ONLY on y != 2 rows.  Re-uses base factories.
    not2 = y != 2
    N_n2 = int(not2.sum())
    print(f"[stage2c] training 6-way stack on y!=2 rows (n={N_n2})...")
    if N_n2 < 50:
        # Pathological subset — skip Stage 2c; hierarchical tier will fail its
        # gate and fall through to Tier 3.
        print(f"[stage2c] SKIP: too few non-class-2 rows ({N_n2} < 50)")
        stage2c_ok = False
        q_oof_n2_cal_full = np.zeros((N, N_CLASSES), dtype=np.float32)
        q_test_n2_cal = np.zeros((len(test), N_CLASSES), dtype=np.float32)
    else:
        stage2c_ok = True
        folds_n2 = folds[not2]
        y_n2 = y[not2]
        w_n2 = inv_freq_weights(y_n2, power=0.7)
        P_oof_n2 = np.zeros((5, N_n2, N_CLASSES), dtype=np.float32)
        P_test_n2 = np.zeros((5, len(test), N_CLASSES), dtype=np.float32)
        for i in range(5):
            Xtr_full_i = base_inputs_tr[i]
            Xte_i = base_inputs_te[i]
            if issparse(Xtr_full_i):
                Xtr_n2 = Xtr_full_i[np.where(not2)[0]]
            else:
                Xtr_n2 = Xtr_full_i[not2]
            for f in range(N_FOLDS):
                tr = folds_n2 != f
                va = folds_n2 == f
                if va.sum() == 0 or tr.sum() == 0:
                    continue
                if issparse(Xtr_n2):
                    Xtr_f = Xtr_n2[np.where(tr)[0]]
                    Xva_f = Xtr_n2[np.where(va)[0]]
                else:
                    Xtr_f = Xtr_n2[tr]
                    Xva_f = Xtr_n2[va]
                try:
                    clf, proba = _safe_fit_predict(
                        lambda fac=base_factories[i]: fac(),
                        Xtr_f, y_n2[tr], Xva_f, sample_weight=w_n2[tr],
                    )
                    P_oof_n2[i, va] = proba
                except Exception as e:
                    print(f"  [stage2c] base {i} fold {f} failed: {e}")
            try:
                clf_full = _fit_full(lambda fac=base_factories[i]: fac(),
                                     Xtr_n2, y_n2, sample_weight=w_n2)
                P_test_n2[i] = _align_proba(clf_full, clf_full.predict_proba(Xte_i))
            except Exception as e:
                print(f"  [stage2c] base {i} full-fit failed: {e}")

        # Meta on not2 rows (re-uses H_sel which we build next — so inline it).
        # Build H_sel first with the same fold-0 chi2 selector used in Stage 3.
        _fold0_tr = folds != 0
        _n_hand_sel = min(20, H_tr.shape[1])
        _sel_H_tmp = SelectKBest(chi2, k=_n_hand_sel).fit(H_tr[_fold0_tr], y[_fold0_tr])
        H_sel_tr_tmp = _sel_H_tmp.transform(H_tr).astype(np.float32)
        H_sel_te_tmp = _sel_H_tmp.transform(H_te).astype(np.float32)

        X_meta_tr_n2 = np.concatenate(
            [P_oof_n2[i] for i in range(5)] + [H_sel_tr_tmp[not2]], axis=1
        )
        X_meta_te_n2 = np.concatenate(
            [P_test_n2[i] for i in range(5)] + [H_sel_te_tmp], axis=1
        )
        q_oof_n2 = np.zeros((N_n2, N_CLASSES), dtype=np.float32)
        for f in range(N_FOLDS):
            tr = folds_n2 != f
            va = folds_n2 == f
            if tr.sum() == 0 or va.sum() == 0:
                continue
            meta_n2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                         n_jobs=-1, random_state=SEED)
            try:
                meta_n2.fit(X_meta_tr_n2[tr], y_n2[tr], sample_weight=w_n2[tr])
            except TypeError:
                meta_n2.fit(X_meta_tr_n2[tr], y_n2[tr])
            q_oof_n2[va] = _align_proba(meta_n2,
                                        meta_n2.predict_proba(X_meta_tr_n2[va]))
        meta_full_n2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                          n_jobs=-1, random_state=SEED)
        try:
            meta_full_n2.fit(X_meta_tr_n2, y_n2, sample_weight=w_n2)
        except TypeError:
            meta_full_n2.fit(X_meta_tr_n2, y_n2)
        q_test_n2 = _align_proba(meta_full_n2,
                                 meta_full_n2.predict_proba(X_meta_te_n2))

        # Calibrate the 6-way head.  Classes with <50 OOF positives get Platt.
        cals_n2 = calibrate_oof(q_oof_n2, y_n2)
        q_oof_n2_cal = apply_calibration(q_oof_n2, cals_n2).astype(np.float32)
        q_test_n2_cal = apply_calibration(q_test_n2, cals_n2).astype(np.float32)

        # Broadcast 6-way OOF into a full-N array aligned with original indices
        # (rows where y == 2 get uniform prior so hierarchical_predict's sub-
        # argmax ignores them — they're routed to class 2 by tau_bin).
        q_oof_n2_cal_full = np.full((N, N_CLASSES), 1.0 / N_CLASSES,
                                    dtype=np.float32)
        q_oof_n2_cal_full[not2] = q_oof_n2_cal

        f1_n2 = f1_score(y_n2, q_oof_n2_cal.argmax(axis=1), average="macro")
        print(f"[stage2c] 6-way head OOF F1: {f1_n2:.4f}")

    # ----------------------------------------------------------------- Stage 3
    print("[stage3] building meta features...")
    # 35 base probs + 20 chi2-selected hand features. Chi2 on OOF-fold-0
    # training portion to avoid selection-bias on validation.
    fold0_tr = folds != 0
    n_hand_sel = min(20, H_tr.shape[1])
    sel_H = SelectKBest(chi2, k=n_hand_sel).fit(H_tr[fold0_tr], y[fold0_tr])
    H_sel_tr = sel_H.transform(H_tr).astype(np.float32)
    H_sel_te = sel_H.transform(H_te).astype(np.float32)

    X_meta_tr = np.concatenate(
        [P_oof[i] for i in range(5)] + [H_sel_tr], axis=1
    )
    X_meta_te = np.concatenate(
        [P_test[i] for i in range(5)] + [H_sel_te], axis=1
    )
    print(f"[stage3] meta dims: {X_meta_tr.shape[1]}")

    # Meta-OOF: same fold assignment (not nested CV).  Fit meta on each fold's
    # training rows and predict on held-out rows.  This reuses base-OOF probs
    # directly — the leakage concern is absent because base probs for fold f
    # were never trained on fold f's rows.
    q_oof = np.zeros((N, N_CLASSES), dtype=np.float32)
    for f in range(N_FOLDS):
        tr_mask = folds != f
        va_mask = folds == f
        meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                  n_jobs=-1, random_state=SEED)
        meta.fit(X_meta_tr[tr_mask], y[tr_mask])
        q_oof[va_mask] = _align_proba(meta, meta.predict_proba(X_meta_tr[va_mask]))
    meta_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                   n_jobs=-1, random_state=SEED)
    meta_full.fit(X_meta_tr, y)
    q_test = _align_proba(meta_full, meta_full.predict_proba(X_meta_te))

    meta_f1 = f1_score(y, q_oof.argmax(axis=1), average="macro")
    print(f"[stage3] meta OOF macro-F1 (raw): {meta_f1:.4f}")

    # ----------------------------------------------------------------- Stage 4
    print("[stage4] per-class isotonic calibration (Platt fallback < 50 pos)")
    cals = calibrate_oof(q_oof, y)
    for k, (kind, _) in enumerate(cals):
        pos = int((y == k).sum())
        print(f"  class {k}: {kind} (n_pos={pos})")
    q_oof_cal = apply_calibration(q_oof, cals)
    q_test_cal = apply_calibration(q_test, cals)

    cal_f1 = f1_score(y, q_oof_cal.argmax(axis=1), average="macro")
    print(f"[stage4] calibrated OOF macro-F1 (argmax): {cal_f1:.4f}")

    # ----------------------------------------------------------------- Stage 5
    # Differential-evolution threshold search, with coord-ascent fallback.
    print("[stage5] tuning per-class thresholds (DE -> coord-ascent fallback)")
    tau_ca_js, tau_ca_f1, tau_ca_raw = tune_tau(q_oof_cal, y)
    print(f"  coord-ascent raw OOF F1: {tau_ca_f1:.4f}")
    try:
        ts = time.time()
        tau_de_js, tau_de_f1, tau_de_raw = tune_tau_de(q_oof_cal, y)
        print(f"  DE raw OOF F1: {tau_de_f1:.4f} ({time.time()-ts:.1f}s)")
    except Exception as e:
        print(f"  DE failed ({e}); using coord-ascent")
        tau_de_f1 = -1.0
        tau_de_js = tau_ca_js
        tau_de_raw = tau_ca_raw

    if tau_de_f1 >= tau_ca_f1:
        tau_js, tau_raw, tau_best_f1 = tau_de_js, tau_de_raw, tau_de_f1
        print(f"[stage5] DE wins -> tau_js {np.array2string(tau_js, precision=3)}")
    else:
        tau_js, tau_raw, tau_best_f1 = tau_ca_js, tau_ca_raw, tau_ca_f1
        print(f"[stage5] coord-ascent wins -> tau_js "
              f"{np.array2string(tau_js, precision=3)}")

    tier_flat_pred_oof = predict_with_tau(q_oof_cal, tau_js)
    tier_flat_pred_test = predict_with_tau(q_test_cal, tau_js)
    tier_flat_f1 = f1_score(y, tier_flat_pred_oof, average="macro")

    # ----------------------------------------------------------------- Stage 6
    print("[stage6] computing rule OOF precisions...")
    masks_tr = compute_rule_masks(train["text_a"].values,
                                  train["achados_a"].values)
    masks_te = compute_rule_masks(test["text_a"].values,
                                  test["achados_a"].values)
    oof_precisions = rule_oof_precisions(masks_tr, tier_flat_pred_oof, y)
    for name, p in oof_precisions.items():
        support = int(masks_tr[name].sum() if name != "R1to2"
                      else (masks_tr["R1to2"] & (tier_flat_pred_oof == 1)).sum())
        print(f"  {name}: precision={p:.3f}  support={support}")

    tier_flat_rules_pred_oof = apply_rules(tier_flat_pred_oof, masks_tr,
                                           oof_precisions)
    tier_flat_rules_pred_test = apply_rules(tier_flat_pred_test, masks_te,
                                            oof_precisions)
    tier_flat_rules_f1 = f1_score(y, tier_flat_rules_pred_oof, average="macro")
    print(f"[stage6] flat+rules OOF macro-F1: {tier_flat_rules_f1:.4f}")
    print(f"[stage6] flat       OOF macro-F1: {tier_flat_f1:.4f}")

    # ----------------------------------------------------------------- Stage 7
    # Hierarchical merge: P(y=2) gate + 6-way argmax.
    if stage2c_ok:
        print("[stage7] tuning hierarchical thresholds...")
        q_oof_n2_cal_routed = q_oof_n2_cal_full  # aligned to full train indices
        ts = time.time()
        tau_bin_best, tau_not2_best, hier_f1_oof = tune_hierarchical(
            P_bin_oof, q_oof_n2_cal_routed, y, n_iter=20,
        )
        print(f"[stage7] hier OOF F1: {hier_f1_oof:.4f} "
              f"(tau_1={tau_bin_best:.3f}, "
              f"tau_not2={np.array2string(tau_not2_best, precision=2)})  "
              f"({time.time()-ts:.1f}s)")

        tier_hier_pred_oof = hierarchical_predict(
            P_bin_oof, q_oof_n2_cal_routed, tau_bin_best, tau_not2_best,
        )
        # For test, stage-2c produced q_test_n2_cal on every test row (size M);
        # hierarchical_predict consumes the full test array directly.
        tier_hier_pred_test = hierarchical_predict(
            P_bin_test, q_test_n2_cal, tau_bin_best, tau_not2_best,
        )
        tier_hier_f1 = f1_score(y, tier_hier_pred_oof, average="macro")
    else:
        print("[stage7] SKIP: stage 2c failed; hierarchical tiers disabled")
        tier_hier_pred_oof = tier_flat_pred_oof.copy()
        tier_hier_pred_test = tier_flat_pred_test.copy()
        tier_hier_f1 = tier_flat_f1
        tau_bin_best = 1.0
        tau_not2_best = np.ones(N_CLASSES)
        hier_f1_oof = tier_flat_f1

    # ----------------------------------------------------------------- Stage 8
    # Pseudo-label refit (one round, class-prior guard). Full-fit only —
    # OOF statistics remain honest.
    pseudo_pred_test_hier = tier_hier_pred_test.copy()
    # Build a "hierarchical-like" calibrated test prob matrix for thresholding:
    # we use q_test_n2_cal (the 6-way, which already excludes class 2) and
    # gate by P_bin_test as below. For the pseudo-threshold we use the
    # 6-way probability directly (max over non-class-2 classes).
    q_test_hier_cal = q_test_n2_cal.copy()
    # Rows likely to be class-2 (P_bin_test high) should not contribute to
    # pseudo-labels; zero their q so max cannot clear threshold.
    route_to_2 = P_bin_test > tau_bin_best
    q_test_hier_cal[route_to_2] = 0.0

    print("[stage8] pseudo-label round (threshold=0.95, no class-2)...")
    idx_pl, y_pl = pseudo_label_round(q_test_hier_cal, threshold=0.95)
    prior_ok, pl_dist = pseudo_prior_ok(y_pl, y, max_ratio=2.0)
    print(f"[stage8] pseudo-labels: n={len(idx_pl)}, dist={pl_dist.tolist()}")

    pseudo_ran = False
    pseudo_pred_test = tier_hier_pred_test.copy()
    if len(idx_pl) < 20:
        print(f"[stage8] SKIP: n={len(idx_pl)} < 20")
    elif not prior_ok:
        print(f"[stage8] SKIP: pseudo class-dist violates >2x train-prior guard")
    else:
        # G8 proxy: measure pseudo-set's predicted class distribution; we already
        # did the deviation check in pseudo_prior_ok. Accept and refit full-fit
        # bases (keep OOF untouched).
        print("[stage8] refitting full-fit bases + meta with pseudo-labels...")
        try:
            stage8_t0 = time.time()
            sel_idx_tr = np.arange(N)
            sel_idx_te = idx_pl
            y_aug = np.concatenate([y, y_pl])
            w_aug = inv_freq_weights(y_aug, power=0.7)

            # Rebuild full-fit base probs on augmented set for test.
            P_test_aug = np.zeros((5, len(test), N_CLASSES), dtype=np.float32)
            for i in range(5):
                Xtr_full_i = base_inputs_tr[i]
                Xte_i = base_inputs_te[i]
                if issparse(Xtr_full_i):
                    X_aug = hstack([Xtr_full_i, Xte_i[sel_idx_te]]).tocsr()
                else:
                    X_aug = np.vstack([Xtr_full_i, Xte_i[sel_idx_te]])
                try:
                    clf_aug = _fit_full(lambda fac=base_factories[i]: fac(),
                                        X_aug, y_aug, sample_weight=w_aug)
                    P_test_aug[i] = _align_proba(clf_aug,
                                                 clf_aug.predict_proba(Xte_i))
                except Exception as e:
                    print(f"  [stage8] base {i} refit failed ({e}); reusing v1")
                    P_test_aug[i] = P_test[i]

            # Refit meta.
            # Re-apply the *same* H_sel (already fit) to train+test rows.
            H_sel_aug = np.vstack([H_sel_tr, H_sel_te[sel_idx_te]])
            X_meta_aug_tr = np.concatenate(
                [np.vstack([P_oof[i], P_test[i][sel_idx_te]]) for i in range(5)]
                + [H_sel_aug], axis=1,
            )
            X_meta_aug_te = np.concatenate(
                [P_test_aug[i] for i in range(5)] + [H_sel_te], axis=1,
            )
            meta_aug = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                          n_jobs=-1, random_state=SEED)
            try:
                meta_aug.fit(X_meta_aug_tr, y_aug, sample_weight=w_aug)
            except TypeError:
                meta_aug.fit(X_meta_aug_tr, y_aug)
            q_test_aug = _align_proba(meta_aug,
                                      meta_aug.predict_proba(X_meta_aug_te))
            q_test_aug_cal = apply_calibration(q_test_aug, cals).astype(np.float32)

            # Also rerun 6-way full-fit on augmented non-class-2 rows.
            if stage2c_ok:
                aug_not2 = y_aug != 2
                aug_not2_idx = np.where(aug_not2)[0]
                if aug_not2_idx.size >= 50:
                    y_aug_n2 = y_aug[aug_not2]
                    w_aug_n2 = inv_freq_weights(y_aug_n2, power=0.7)
                    P_test_aug_n2 = np.zeros((5, len(test), N_CLASSES),
                                             dtype=np.float32)
                    for i in range(5):
                        Xtr_full_i = base_inputs_tr[i]
                        Xte_i = base_inputs_te[i]
                        if issparse(Xtr_full_i):
                            X_all = hstack([Xtr_full_i, Xte_i[sel_idx_te]]).tocsr()
                            X_all_n2 = X_all[aug_not2_idx]
                        else:
                            X_all = np.vstack([Xtr_full_i, Xte_i[sel_idx_te]])
                            X_all_n2 = X_all[aug_not2]
                        try:
                            clf_aug_n2 = _fit_full(
                                lambda fac=base_factories[i]: fac(),
                                X_all_n2, y_aug_n2, sample_weight=w_aug_n2,
                            )
                            P_test_aug_n2[i] = _align_proba(
                                clf_aug_n2, clf_aug_n2.predict_proba(Xte_i))
                        except Exception as e:
                            print(f"  [stage8] n2 base {i} refit failed ({e})")
                            P_test_aug_n2[i] = P_test_n2[i]
                    # meta on not2
                    X_meta_aug_n2_tr = np.concatenate(
                        [np.vstack([P_oof_n2[i],
                                    P_test_n2[i][sel_idx_te][y_pl != 2]])
                         for i in range(5)]
                        + [np.vstack([H_sel_tr[not2],
                                      H_sel_te[sel_idx_te][y_pl != 2]])],
                        axis=1,
                    )
                    X_meta_aug_n2_te = np.concatenate(
                        [P_test_aug_n2[i] for i in range(5)] + [H_sel_te],
                        axis=1,
                    )
                    meta_aug_n2 = LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=2000,
                        n_jobs=-1, random_state=SEED,
                    )
                    try:
                        meta_aug_n2.fit(X_meta_aug_n2_tr, y_aug_n2,
                                        sample_weight=w_aug_n2)
                    except TypeError:
                        meta_aug_n2.fit(X_meta_aug_n2_tr, y_aug_n2)
                    q_test_aug_n2 = _align_proba(
                        meta_aug_n2, meta_aug_n2.predict_proba(X_meta_aug_n2_te))
                    q_test_aug_n2_cal = apply_calibration(
                        q_test_aug_n2, cals_n2).astype(np.float32)

                    # Rebuild binary head on augmented (treat pseudo class 2
                    # irrelevant since no class-2 pseudo rows).
                    y_bin_aug = (y_aug == 2).astype(int)
                    if issparse(M1_tr_full):
                        M1_aug = hstack([M1_tr_full, M1_te_full[sel_idx_te]]).tocsr()
                    else:
                        M1_aug = np.vstack([M1_tr_full, M1_te_full[sel_idx_te]])
                    clf_bin_aug = LogisticRegression(
                        C=2.0, class_weight="balanced", solver="lbfgs",
                        max_iter=2000, n_jobs=-1, random_state=SEED,
                    )
                    clf_bin_aug.fit(M1_aug, y_bin_aug)
                    cls_aug = list(clf_bin_aug.classes_)
                    P_bin_test_aug = (clf_bin_aug.predict_proba(M1_te_full)[
                        :, cls_aug.index(1)] if 1 in cls_aug
                        else np.zeros(len(test), dtype=np.float32))

                    pseudo_pred_test = hierarchical_predict(
                        P_bin_test_aug, q_test_aug_n2_cal,
                        tau_bin_best, tau_not2_best,
                    )
                    pseudo_ran = True
                    print(f"[stage8] refit complete ({time.time()-stage8_t0:.1f}s)")
                else:
                    print("[stage8] SKIP: augmented not2 too small")
            else:
                # Without stage 2c, fall back to flat tau argmax on q_test_aug_cal.
                pseudo_pred_test = predict_with_tau(q_test_aug_cal, tau_js)
                pseudo_ran = True
                print(f"[stage8] flat refit complete ({time.time()-stage8_t0:.1f}s)")

            if (time.time() - stage8_t0) > 20 * 60:
                print("[stage8] WARN: exceeded 20-minute budget")
        except Exception as e:
            print(f"[stage8] ERROR: {e}; skipping pseudo-label step")
            pseudo_pred_test = tier_hier_pred_test.copy()
            pseudo_ran = False

    # ----------------------------------------------------------------- Stage 9
    # Lookup tie-breaker, applied ONLY for predicted class in {1,2} with
    # top prob < 0.85, and only when stage-2c provided calibrated probs.
    print("[stage9] lookup tie-breaker (exact MD5 + Jaccard>=0.97)...")
    # Use q_test_n2_cal as the top-prob gate (it's the 6-way calibrated; for
    # rows routed to 2 via P_bin we re-use P_bin_test as an implicit prob).
    topp_test = np.maximum(q_test_n2_cal.max(axis=1), P_bin_test).astype(np.float32)
    base_for_lookup = pseudo_pred_test if pseudo_ran else tier_hier_pred_test
    # Build a lightweight topp matrix just for the lookup gate.
    topp_mat = np.zeros((len(test), N_CLASSES), dtype=np.float32)
    topp_mat[np.arange(len(test)), 0] = topp_test  # slot 0 arbitrary; only .max used
    lookup_pred_test, n_flips = apply_lookup_tiebreak(
        test, base_for_lookup, topp_mat, train, tau_j=0.97,
    )
    print(f"[stage9] lookup flips: {n_flips}")

    # Simulate G9 over OOF (leave-one-MD5-group-out) on the hierarchical OOF.
    q_oof_for_gate = np.zeros((N, N_CLASSES), dtype=np.float32)
    if stage2c_ok:
        q_oof_for_gate[np.arange(N), 0] = np.maximum(
            q_oof_n2_cal_full.max(axis=1), P_bin_oof,
        )
    else:
        q_oof_for_gate[np.arange(N), 0] = q_oof_cal.max(axis=1)
    lookup_oof_pred, oof_flips = apply_lookup_tiebreak_oof(
        train, tier_hier_pred_oof, q_oof_for_gate, tau_j=0.97,
    )
    hier_lookup_f1_oof = f1_score(y, lookup_oof_pred, average="macro")

    # ----------------------------------------------------------------- Stage 11
    # Baseline 5-fold CV (probability outputs for blending).
    print("[stage11] baseline 5-fold CV + full-fit for test...")
    s11_t0 = time.time()
    P_baseline_oof = np.zeros((N, N_CLASSES), dtype=np.float32)
    bin0_oof = np.zeros(N, dtype=np.float32)
    for f in range(N_FOLDS):
        tr_mask = folds != f
        va_mask = folds == f
        if tr_mask.sum() < 10 or va_mask.sum() == 0:
            continue
        try:
            probs_va, bin0_va, _ = baseline_predict_proba(
                train.iloc[tr_mask.nonzero()[0]].reset_index(drop=True),
                train.iloc[va_mask.nonzero()[0]].reset_index(drop=True),
                y[tr_mask],
            )
            P_baseline_oof[va_mask] = probs_va
            bin0_oof[va_mask] = bin0_va
        except Exception as e:
            print(f"  [stage11] fold {f} failed ({e}); using uniform probs")
            P_baseline_oof[va_mask] = 1.0 / N_CLASSES
            bin0_oof[va_mask] = 0.0
    try:
        P_baseline_test, bin0_test_base, _ = baseline_predict_proba(train, test, y)
    except Exception as e:
        print(f"  [stage11] full fit failed ({e}); uniform fallback")
        P_baseline_test = np.full((len(test), N_CLASSES), 1.0 / N_CLASSES,
                                  dtype=np.float32)
        bin0_test_base = np.zeros(len(test), dtype=np.float32)

    baseline_oof_argmax = np.argmax(P_baseline_oof, axis=1)
    baseline_oof_f1_argmax = f1_score(y, baseline_oof_argmax, average="macro")
    baseline_oof_pred_thresh = baseline_predict_final(
        P_baseline_oof, bin0_oof, train["report"].values,
    )
    baseline_oof_f1_thresh = f1_score(y, baseline_oof_pred_thresh, average="macro")
    baseline_pred_test = baseline_predict_final(
        P_baseline_test, bin0_test_base, test["report"].values,
    )
    print(f"[stage11] baseline OOF F1 (argmax): {baseline_oof_f1_argmax:.4f}")
    print(f"[stage11] baseline OOF F1 (with thresholds): "
          f"{baseline_oof_f1_thresh:.4f}  ({time.time()-s11_t0:.1f}s)")

    # ----------------------------------------------------------------- Stage 12
    # Build full-N hierarchical calibrated prob matrices for OOF + test.
    def _build_hier_probs(p_bin, q_n2_cal, tau_bin, tau_not2):
        n = p_bin.shape[0]
        out = np.zeros((n, N_CLASSES), dtype=np.float32)
        route2 = p_bin > tau_bin
        out[route2, 2] = 1.0
        other = ~route2
        if other.any():
            tau_safe = np.where(tau_not2 > 1e-6, tau_not2, 1.0).astype(np.float32)
            z = q_n2_cal[other] / tau_safe
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            e_sum = e.sum(axis=1, keepdims=True)
            e_sum = np.where(e_sum > 0, e_sum, 1.0)
            out[other] = (e / e_sum).astype(np.float32)
        return out

    hier_probs_oof = _build_hier_probs(
        P_bin_oof, q_oof_n2_cal_full, tau_bin_best, tau_not2_best,
    )
    hier_probs_test = _build_hier_probs(
        P_bin_test, q_test_n2_cal, tau_bin_best, tau_not2_best,
    )

    print("[stage12] probability-space blending scan...")
    blend_scores = []
    for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
        P_blend_oof = alpha * P_baseline_oof + (1 - alpha) * hier_probs_oof
        pred_blend_oof = baseline_predict_final(
            P_blend_oof, bin0_oof, train["report"].values,
        )
        f1_blend = f1_score(y, pred_blend_oof, average="macro")
        blend_scores.append((alpha, f1_blend, pred_blend_oof))
        print(f"[stage12] blend alpha={alpha:.2f}: OOF F1 = {f1_blend:.4f}")
    alpha_best, blend_best_oof_f1, pred_blend_oof_best = max(
        blend_scores, key=lambda x: x[1],
    )
    P_blend_test = alpha_best * P_baseline_test + (1 - alpha_best) * hier_probs_test
    pred_blend_test = baseline_predict_final(
        P_blend_test, bin0_test_base, test["report"].values,
    )
    print(f"[stage12] best alpha={alpha_best:.2f} OOF F1={blend_best_oof_f1:.4f}")

    # ----------------------------------------------------------------- Per-class quality floor
    def per_class_sane(pred_oof, y_true):
        per_class = [f1_score((y_true == k).astype(int),
                              (pred_oof == k).astype(int),
                              zero_division=0) for k in range(N_CLASSES)]
        return all(
            (per_class[k] >= 0.40) if k != 5 else (per_class[k] >= 0.30)
            for k in range(N_CLASSES) if (y_true == k).sum() > 0
        ), per_class

    hier_oof_f1 = tier_hier_f1
    flat_rules_oof_f1 = tier_flat_rules_f1

    baseline_sane, baseline_pc = per_class_sane(baseline_oof_pred_thresh, y)
    hier_sane, hier_pc = per_class_sane(tier_hier_pred_oof, y)
    flat_sane, flat_pc = per_class_sane(tier_flat_rules_pred_oof, y)
    blend_sane, blend_pc = per_class_sane(pred_blend_oof_best, y)

    print("\n[v3] Candidate OOF F1s:")
    print(f"  Baseline:    {baseline_oof_f1_thresh:.4f}  per-class "
          f"{[f'{v:.2f}' for v in baseline_pc]}  sane={baseline_sane}")
    print(f"  Hier:        {hier_oof_f1:.4f}  per-class "
          f"{[f'{v:.2f}' for v in hier_pc]}  sane={hier_sane}")
    print(f"  FlatRules:   {flat_rules_oof_f1:.4f}  per-class "
          f"{[f'{v:.2f}' for v in flat_pc]}  sane={flat_sane}")
    print(f"  Blend@{alpha_best:.2f}:  {blend_best_oof_f1:.4f}  per-class "
          f"{[f'{v:.2f}' for v in blend_pc]}  sane={blend_sane}")

    all_cands = [
        ("Baseline",   baseline_oof_f1_thresh, baseline_pred_test, baseline_sane),
        ("Hier",       hier_oof_f1,            tier_hier_pred_test, hier_sane),
        ("FlatRules",  flat_rules_oof_f1,      tier_flat_rules_pred_test, flat_sane),
        (f"Blend@{alpha_best:.2f}", blend_best_oof_f1, pred_blend_test, blend_sane),
    ]
    candidates = [(n, f, p) for (n, f, p, s) in all_cands if s]

    if not candidates:
        print("[v3] No candidate passes G4_per_class_sane -> emergency baseline")
        final_name, final_pred = "Baseline_emergency", baseline_pred_test
        final_f1 = baseline_oof_f1_thresh
    else:
        final_name, final_f1, final_pred = max(candidates, key=lambda x: x[1])
    print(f"[v3] Selected: {final_name} (F1={final_f1:.4f})")

    # ----------------------------------------------------------------- Submission
    submission = pd.DataFrame({"ID": test["ID"].values,
                               "target": final_pred.astype(int)})
    submission.to_csv("submission.csv", index=False)
    print("\n===== SUBMISSION SUMMARY =====")
    print(f"Selected: {final_name}")
    print("Prediction distribution:")
    print(pd.Series(final_pred).value_counts().sort_index().to_string())
    print(f"\nWrote submission.csv ({len(submission)} rows)")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Baseline (v2_Tier4): reproduce the 0.80972 multihead_thresh_tuned baseline
# Split into proba + final so v3 can blend baseline probs with hierarchical.
# ---------------------------------------------------------------------------

def _baseline_clean_achados(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    m = re.search(r"achados:(.*?)(análise comparativa:|$)", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"[0-9]+,[0-9]+", "NUM", s)
    s = re.sub(r"[0-9]+", "NUM", s)
    return s


def _baseline_clean_full(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[0-9]+,[0-9]+", "NUM", s)
    s = re.sub(r"[0-9]+", "NUM", s)
    return s


def _baseline_dense(df):
    t = df["report"].fillna("").astype(str).str.lower()
    feats = pd.DataFrame({
        "report_length": t.apply(len),
        "has_measurement": t.str.contains(r"\b(?:cm|mm|medindo)\b", regex=True).astype(int),
        "has_spiculation": t.str.contains(r"espiculad", regex=True).astype(int),
        "has_distortion": t.str.contains(r"distorção arquitetural", regex=True).astype(int),
        "has_biopsy": t.str.contains(r"biopsy|biópsia|resultado de cine|carcinoma", regex=True).astype(int),
    })
    return csr_matrix(feats.values)


def baseline_predict_proba(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           y: np.ndarray):
    """Fit baseline SVC ensemble + LGB + class-0 binary detector.

    Returns (probs_test[N_test,7], bin0_test[N_test], meta_dict). Stops before
    threshold overrides so the raw probability matrix can be blended.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC as LSVC
    from sklearn.calibration import CalibratedClassifierCV as CCV
    import lightgbm as lgb

    tr = train_df.copy()
    te = test_df.copy()
    tr["_ach"] = tr["report"].apply(_baseline_clean_achados)
    tr["_full"] = tr["report"].apply(_baseline_clean_full)
    te["_ach"] = te["report"].apply(_baseline_clean_achados)
    te["_full"] = te["report"].apply(_baseline_clean_full)

    Xd_tr, Xd_te = _baseline_dense(tr), _baseline_dense(te)
    tfA = FeatureUnion([
        ("word", TfIdf(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
        ("char", TfIdf(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                       max_df=0.95, sublinear_tf=True, max_features=80000)),
    ])
    Xa_tr = tfA.fit_transform(tr["_ach"])
    Xa_te = tfA.transform(te["_ach"])

    tfF = FeatureUnion([
        ("word", TfIdf(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
        ("char", TfIdf(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                       max_df=0.95, sublinear_tf=True, max_features=80000)),
    ])
    Xf_tr = tfF.fit_transform(tr["_full"])
    Xf_te = tfF.transform(te["_full"])

    tfF2 = FeatureUnion([
        ("word", TfIdf(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)),
        ("char", TfIdf(analyzer="char_wb", ngram_range=(3, 6), min_df=3,
                       max_df=0.95, sublinear_tf=True, max_features=100000)),
    ])
    Xf2_tr = tfF2.fit_transform(tr["_full"])
    Xf2_te = tfF2.transform(te["_full"])

    lgbX_tr = hstack([Xa_tr, Xd_tr]).tocsr()
    lgbX_te = hstack([Xa_te, Xd_te]).tocsr()

    def svc():
        return CCV(LSVC(class_weight="balanced", random_state=SEED, max_iter=10000),
                   cv=3, method="sigmoid")

    def _align(clf, P):
        out = np.zeros((P.shape[0], N_CLASSES), dtype=np.float32)
        for j, c in enumerate(np.asarray(clf.classes_, dtype=int)):
            if 0 <= c < N_CLASSES:
                out[:, c] = P[:, j]
        return out

    sA = svc(); sA.fit(Xa_tr, y); pA = _align(sA, sA.predict_proba(Xa_te))
    sF = svc(); sF.fit(Xf_tr, y); pF = _align(sF, sF.predict_proba(Xf_te))
    sF2 = svc(); sF2.fit(Xf2_tr, y); pF2 = _align(sF2, sF2.predict_proba(Xf2_te))
    lgb_model = lgb.LGBMClassifier(class_weight="balanced", n_estimators=300,
                                   learning_rate=0.05, max_depth=6,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_model.fit(lgbX_tr, y)
    pL = _align(lgb_model, lgb_model.predict_proba(lgbX_te))
    svc_ens = 0.25 * pA + 0.40 * pF + 0.35 * pF2
    ens = (0.70 * svc_ens + 0.30 * pL).astype(np.float32)

    # Binary class-0 detector
    mask02 = np.isin(y, [0, 2])
    if mask02.sum() >= 2 and len(np.unique(y[mask02])) == 2:
        y02 = (y[mask02] == 0).astype(int)
        bin0 = svc(); bin0.fit(Xf_tr[mask02], y02)
        if 1 in bin0.classes_:
            bin0_te = bin0.predict_proba(Xf_te)[:, list(bin0.classes_).index(1)]
        else:
            bin0_te = np.zeros(len(te), dtype=np.float32)
    else:
        bin0_te = np.zeros(len(te), dtype=np.float32)

    return ens, bin0_te.astype(np.float32), {"n_train": len(tr), "n_test": len(te)}


def baseline_predict_final(probs_test: np.ndarray, bin0_test: np.ndarray,
                           text_values) -> np.ndarray:
    """Apply baseline threshold overrides and safe_rule regex to produce final preds."""
    ens = probs_test
    preds = np.argmax(ens, axis=1).copy()
    preds[ens[:, 6] > 0.10] = 6
    preds[(ens[:, 5] > 0.15) & (preds != 6)] = 5
    preds[(ens[:, 4] > 0.23) & (preds != 6) & (preds != 5)] = 4
    preds[(ens[:, 3] > 0.38) & (preds != 6) & (preds != 5) & (preds != 4)] = 3
    preds[(bin0_test > 0.55) & (preds == 2)] = 0

    def safe_rule(text, pred):
        low = str(text).lower()
        if re.search(r"resultado de cine grau 3|carcinoma|\bcdis\b", low):
            return 6
        if "espiculad" in low and "distorção" in low and pred < 4:
            return 5
        return int(pred)

    preds = np.array([safe_rule(t, p) for t, p in zip(text_values, preds)],
                     dtype=int)
    return preds


def run_baseline_fallback(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          y: np.ndarray) -> tuple[str, np.ndarray]:
    """v2_Tier4 safety net: full baseline using proba + final split."""
    probs, bin0, _ = baseline_predict_proba(train_df, test_df, y)
    preds = baseline_predict_final(probs, bin0, test_df["report"].values)
    return "v2_Tier4", preds


if __name__ == "__main__":
    main()
