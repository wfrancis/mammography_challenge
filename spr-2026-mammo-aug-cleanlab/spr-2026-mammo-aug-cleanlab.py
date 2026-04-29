from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


COMPETITION_NAME = "spr-2026-mammography-report-classification"
DEFAULT_COMPETITION_PATH = Path("/kaggle/input") / COMPETITION_NAME
LOCAL_FALLBACK_PATH = Path(__file__).resolve().parents[2] / "data" / "raw"

# --- Augmentation + cleanlab fixes (added for 0.80+ attempt) ---
import re as _re_aug
import random as _random_aug

CLEANLAB_FIXES = [
    (7937, 2),
    (7587, 4),
    (6741, 2),
    (3705, 0),
    (13000, 2),
    (15197, 4),
    (15677, 2),
    (4659, 6),
    (18152, 4),
    (3650, 0),
    (7392, 2),
    (4193, 0),
    (4586, 4),
    (12523, 2),
    (7267, 4),
    (16254, 2),
    (10700, 4),
    (3213, 4),
    (15193, 0),
    (16636, 4),
    (3757, 4),
    (15852, 1),
    (13603, 1),
    (8799, 2),
    (438, 0),
    (14535, 1),
    (3726, 0),
    (2719, 1),
    (8108, 0),
    (16867, 2),
    (17801, 2),
    (14484, 1),
    (15778, 4),
    (1741, 2),
    (14551, 0),
    (1752, 0),
    (368, 4),
    (14621, 1),
    (2291, 0),
    (10304, 2),
    (5898, 0),
    (15854, 1),
    (7077, 4),
    (15370, 4),
    (9293, 4),
    (16200, 1),
    (15527, 0),
    (15700, 0),
    (3546, 0),
    (4083, 0),
    (6272, 0),
    (5909, 0),
    (17679, 1),
    (528, 0),
    (3673, 0),
    (13481, 0),
    (11080, 4),
    (11747, 1),
    (100, 4),
    (13389, 2),
    (4097, 0),
    (17619, 4),
    (8249, 0),
    (147, 4),
    (2351, 4),
    (17203, 1),
    (1310, 4),
    (17285, 0),
    (4383, 0),
    (7911, 0),
    (11573, 0),
    (4942, 0),
    (5945, 0),
    (10272, 0),
    (470, 0),
    (7803, 1),
    (4119, 4),
    (4595, 0),
    (3854, 4),
    (3982, 0),
    (1010, 6),
    (2616, 0),
    (961, 0),
    (6339, 2),
    (7516, 0),
    (4739, 6),
    (1846, 4),
    (2574, 0),
    (5023, 4),
    (14935, 4),
    (3980, 2),
    (8452, 0),
    (14802, 1),
    (10862, 4),
    (3286, 4),
    (10534, 0),
    (12232, 0),
    (12476, 0),
    (9836, 0),
    (14687, 0),
    (6900, 4),
    (5840, 4),
    (293, 4),
    (15209, 2),
    (6430, 6),
    (5500, 0),
    (2531, 0),
    (6066, 0),
    (3973, 0),
    (6277, 0),
    (18083, 4),
    (1168, 0),
    (7703, 5),
    (13866, 0),
    (15650, 0),
    (5915, 0),
    (10444, 0),
    (16810, 4),
    (5870, 0),
    (501, 0),
    (1549, 0),
    (8556, 4),
    (8907, 2),
    (201, 4),
    (12488, 4),
    (16974, 4),
    (9526, 0),
    (5503, 4),
    (8622, 0),
    (15640, 0),
    (10437, 5),
    (8241, 4),
    (8378, 4),
    (9691, 4),
    (12789, 5),
    (11450, 0),
    (2410, 0),
    (8123, 0),
    (12245, 0),
    (16773, 5),
    (4988, 0),
    (7008, 0),
    (15647, 0),
    (237, 0),
    (1355, 0),
    (33, 0),
    (15138, 0),
    (4006, 0),
    (7126, 5),
    (10717, 0),
    (10337, 4),
    (10455, 4),
    (8202, 4),
    (3595, 0),
    (5346, 0),
    (5941, 0),
    (2454, 0),
    (315, 0),
    (5439, 4),
    (12475, 0),
    (11537, 0),
    (17650, 0),
    (7621, 0),
    (13712, 5),
    (6268, 5),
    (17421, 0),
    (16721, 4),
    (9083, 5),
    (11178, 5),
    (2541, 0),
    (18054, 4),
    (8926, 4),
    (12275, 4),
    (9041, 5),
    (10691, 5),
    (10141, 5),
    (6987, 4)
]

AUG_FACTORS = {0: 2, 3: 2, 4: 4, 5: 15, 6: 8}

def augment_report(text, n_variants, seed=2026, drop_p=0.10, word_drop=0.05):
    rng = _random_aug.Random(int(seed))
    sents = _re_aug.split(r"(?<=[.!?])\s+|\n+", str(text))
    sents = [s for s in sents if s.strip()]
    out = []
    for _ in range(n_variants):
        kept = [s for s in sents if rng.random() > drop_p] or sents[:1]
        if len(kept) > 2 and rng.random() < 0.3:
            middle = kept[1:-1]
            rng.shuffle(middle)
            kept = [kept[0]] + middle + [kept[-1]]
        words = [w for w in " ".join(kept).split() if rng.random() > word_drop]
        out.append(" ".join(words))
    return out

SPARSE_FOLD_SEEDS = [2026, 2031, 2027, 2053, 2077, 2099, 2143, 2203]
CLASSWISE_WEIGHTS = np.asarray(
    [
        [0.04905408248305321, 0.005982762668281794, 0.020572368055582047, 0.007366769481450319, 0.012082504108548164, 0.0024922965094447136, 0.003024952718988061],
        [0.04457085207104683, 0.10379934310913086, 0.08265554904937744, 0.09272751212120056, 0.06884706020355225, 0.10265574604272842, 0.030428024008870125],
        [0.09487763047218323, 0.17614243924617767, 0.15265023708343506, 0.14451181888580322, 0.15247108042240143, 0.1177394762635231, 0.156009241938591],
        [0.10192140191793442, 0.11662568151950836, 0.09571439772844315, 0.09344692528247833, 0.07372507452964783, 0.12290634959936142, 0.12284129858016968],
        [0.2700907588005066, 0.21500660479068756, 0.21368379890918732, 0.18973004817962646, 0.24886639416217804, 0.21702364087104797, 0.2203606218099594],
        [0.20549196004867554, 0.14275763928890228, 0.1646636724472046, 0.1398501694202423, 0.1610427051782608, 0.15523293614387512, 0.16500380635261536],
        [0.21724802255630493, 0.21115314960479736, 0.23513057827949524, 0.22529804706573486, 0.22085288166999817, 0.23210303485393524, 0.2568018436431885],
        [0.0038928112480789423, 0.018937906250357628, 0.011500696651637554, 0.08245238661766052, 0.016520075500011444, 0.010681288316845894, 0.023605847731232643],
        [0.011287088505923748, 0.003536567324772477, 0.02332824096083641, 0.0021838792599737644, 0.041111718863248825, 0.027822602540254593, 0.013918716460466385],
        [0.0015653740847483277, 0.006057902239263058, 0.00010047085379483178, 0.022432446479797363, 0.00448050070554018, 0.011342626065015793, 0.008005654439330101],
    ],
    dtype=np.float32,
)
OFFSETS = np.asarray(
    [-0.08280894160270691, -0.17869539558887482, 0.19871781766414642, -0.0925980880856514, -0.06091325357556343, 0.10388537496328354, 0.05476764217019081],
    dtype=np.float32,
)


def resolve_data_root() -> Path:
    if (DEFAULT_COMPETITION_PATH / "train.csv").exists() and (DEFAULT_COMPETITION_PATH / "test.csv").exists():
        return DEFAULT_COMPETITION_PATH

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_path in kaggle_input.rglob("train.csv"):
            candidate = train_path.parent
            if (candidate / "test.csv").exists():
                return candidate

    if (LOCAL_FALLBACK_PATH / "train.csv").exists() and (LOCAL_FALLBACK_PATH / "test.csv").exists():
        return LOCAL_FALLBACK_PATH

    raise FileNotFoundError("Could not locate train.csv and test.csv under /kaggle/input or the local fallback.")


def build_features(train_text: pd.Series, test_text: pd.Series):
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(3, 5),
        min_df=2,
        max_features=180000,
        sublinear_tf=True,
    )

    x_train = hstack(
        [
            word_vectorizer.fit_transform(train_text),
            char_vectorizer.fit_transform(train_text),
        ],
        format="csr",
    )
    x_test = hstack(
        [
            word_vectorizer.transform(test_text),
            char_vectorizer.transform(test_text),
        ],
        format="csr",
    )
    return x_train, x_test


def make_sparse_model():
    base = LogisticRegression(
        C=3.0,
        max_iter=400,
        class_weight="balanced",
        solver="liblinear",
        random_state=2026,
    )
    return OneVsRestClassifier(base, n_jobs=1)


def make_rule_specialist(texts: pd.Series, labels: list[int]) -> np.ndarray:
    text = texts.fillna("").astype(str).str.lower()
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    rule = np.full((len(texts), len(labels)), 1e-6, dtype=np.float32)

    rule[:, label_to_index[2]] = 0.55
    rule[:, label_to_index[1]] = 0.15
    rule[:, label_to_index[0]] = 0.10
    rule[:, label_to_index[3]] = 0.10
    rule[:, label_to_index[4]] = 0.05
    rule[:, label_to_index[5]] = 0.025
    rule[:, label_to_index[6]] = 0.025

    path = (
        text.str.contains("biops", regex=False)
        | text.str.contains("carcin", regex=False)
        | text.str.contains("malign", regex=False)
    )
    susp5 = (text.str.contains("espicul", regex=False) | text.str.contains("retra", regex=False)) & ~path
    calc4 = (
        text.str.contains("amorf", regex=False)
        | text.str.contains("pleomorf", regex=False)
        | text.str.contains("segmentar", regex=False)
    ) & ~path & ~susp5
    stable3 = (
        text.str.contains("estável", regex=False)
        | text.str.contains("estavel", regex=False)
        | text.str.contains("controle", regex=False)
    ) & ~path & ~susp5 & ~calc4
    recall0 = (
        text.str.contains("compress", regex=False)
        | text.str.contains("magnific", regex=False)
        | text.str.contains("reavalia", regex=False)
    ) & ~path & ~susp5 & ~calc4 & ~stable3
    normal1 = text.str.contains("sem alterações significativas", regex=False) & ~path & ~susp5 & ~calc4 & ~stable3 & ~recall0

    for mask, label, confidence in [
        (path, 6, 0.97),
        (susp5, 5, 0.93),
        (calc4, 4, 0.90),
        (stable3, 3, 0.88),
        (recall0, 0, 0.85),
        (normal1, 1, 0.80),
    ]:
        idx = np.where(mask.to_numpy())[0]
        rule[idx] = 1e-6
        rule[idx, label_to_index[label]] = confidence
        remain = (1.0 - confidence) / (len(labels) - 1)
        for class_idx in range(len(labels)):
            if class_idx != label_to_index[label]:
                rule[idx, class_idx] = remain

    return rule


def build_sparse_fold_bag(train_text: pd.Series, train_y: np.ndarray, test_text: pd.Series, cv_seed: int) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    test_proba = np.zeros((len(test_text), len(np.unique(train_y))), dtype=np.float64)
    for train_idx, _ in splitter.split(train_text, train_y):
        # Augment rare-class samples in this training fold
        aug_texts = list(train_text.iloc[train_idx].values)
        aug_y = list(train_y[train_idx])
        for idx_in_fold, orig_idx in enumerate(train_idx):
            cls = int(train_y[orig_idx])
            if cls in AUG_FACTORS:
                variants = augment_report(
                    train_text.iloc[orig_idx],
                    AUG_FACTORS[cls],
                    seed=cv_seed * 10000 + int(orig_idx),
                )
                aug_texts.extend(variants)
                aug_y.extend([cls] * len(variants))
        aug_y = np.array(aug_y)
        aug_text_series = pd.Series(aug_texts)
        x_train, x_test = build_features(aug_text_series, test_text)
        model = make_sparse_model()
        model.fit(x_train, aug_y)
        test_proba += model.predict_proba(x_test) / splitter.get_n_splits()
    return test_proba


def build_hierarchy_component(
    train_text: pd.Series,
    train_labels: np.ndarray,
    test_text: pd.Series,
    labels: list[int],
    rule_test_proba: np.ndarray,
) -> np.ndarray:
    group_map = {0: 1, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    group_y = np.array([group_map[int(label)] for label in train_labels], dtype=np.int64)

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    test_proba = np.zeros((len(test_text), len(labels)), dtype=np.float64)

    for train_idx, _ in splitter.split(train_text, group_y):
        fold_text = train_text.iloc[train_idx]
        fold_labels = train_labels[train_idx]
        fold_groups = group_y[train_idx]

        x_train, x_test = build_features(fold_text, test_text)

        route_model = LogisticRegression(
            C=3.0,
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            random_state=2026,
        )
        route_model.fit(x_train, fold_groups)
        route_proba = route_model.predict_proba(x_test)

        fold_component = np.zeros((len(test_text), len(labels)), dtype=np.float64)

        benign_mask = np.isin(fold_labels, [1, 2])
        benign_model = LogisticRegression(
            C=3.0,
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            random_state=2026,
        )
        benign_model.fit(x_train[benign_mask], fold_labels[benign_mask])
        benign_proba = benign_model.predict_proba(x_test)
        fold_component[:, label_to_index[1]] = route_proba[:, 0] * benign_proba[:, list(benign_model.classes_).index(1)]
        fold_component[:, label_to_index[2]] = route_proba[:, 0] * benign_proba[:, list(benign_model.classes_).index(2)]

        callback_mask = np.isin(fold_labels, [0, 3])
        callback_model = LogisticRegression(
            C=3.0,
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            random_state=2026,
        )
        callback_model.fit(x_train[callback_mask], fold_labels[callback_mask])
        callback_proba = callback_model.predict_proba(x_test)
        fold_component[:, label_to_index[0]] = route_proba[:, 1] * callback_proba[:, list(callback_model.classes_).index(0)]
        fold_component[:, label_to_index[3]] = route_proba[:, 1] * callback_proba[:, list(callback_model.classes_).index(3)]

        suspicious_mask = np.isin(fold_labels, [4, 5, 6])
        suspicious_model = LogisticRegression(
            C=3.0,
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            random_state=2026,
        )
        suspicious_model.fit(x_train[suspicious_mask], fold_labels[suspicious_mask])
        suspicious_proba = suspicious_model.predict_proba(x_test)
        fold_component[:, label_to_index[4]] = route_proba[:, 2] * suspicious_proba[:, list(suspicious_model.classes_).index(4)]
        fold_component[:, label_to_index[5]] = route_proba[:, 2] * suspicious_proba[:, list(suspicious_model.classes_).index(5)]
        fold_component[:, label_to_index[6]] = route_proba[:, 2] * suspicious_proba[:, list(suspicious_model.classes_).index(6)]

        fold_component = np.clip(fold_component, 1e-9, None)
        fold_component /= fold_component.sum(axis=1, keepdims=True)
        fold_component = 0.9 * fold_component + 0.1 * rule_test_proba
        test_proba += fold_component / splitter.get_n_splits()

    return test_proba


def main():
    data_root = resolve_data_root()
    print(f"Using data root: {data_root}")
    train_df = pd.read_csv(data_root / "train.csv")
    test_df = pd.read_csv(data_root / "test.csv")

    labels = sorted(train_df["target"].unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_train = train_df["target"].map(label_to_index).to_numpy()
    # Apply cleanlab label fixes
    for row_idx, to_label in CLEANLAB_FIXES:
        if row_idx < len(y_train):
            y_train[row_idx] = label_to_index[to_label]
    print(f"Applied {len(CLEANLAB_FIXES)} cleanlab fixes")
    train_text = train_df["report"].fillna("").astype(str)
    test_text = test_df["report"].fillna("").astype(str)

    rule_test_proba = make_rule_specialist(test_df["report"], labels)
    # Also apply cleanlab fixes to hierarchy labels
    train_labels_fixed = train_df["target"].to_numpy().copy()
    for row_idx, to_label in CLEANLAB_FIXES:
        if row_idx < len(train_labels_fixed):
            train_labels_fixed[row_idx] = to_label
    hierarchy_test_proba = build_hierarchy_component(
        train_text,
        train_labels_fixed,
        test_text,
        labels,
        rule_test_proba,
    )

    sparse_seed_to_proba = {}
    for fold_seed in SPARSE_FOLD_SEEDS:
        sparse_seed_to_proba[fold_seed] = build_sparse_fold_bag(train_text, y_train, test_text, fold_seed)
        print(f"Finished sparse seed {fold_seed}")

    ordered_components = [
        sparse_seed_to_proba[2026],
        rule_test_proba,
        hierarchy_test_proba,
        sparse_seed_to_proba[2031],
        sparse_seed_to_proba[2027],
        sparse_seed_to_proba[2053],
        sparse_seed_to_proba[2077],
        sparse_seed_to_proba[2099],
        sparse_seed_to_proba[2143],
        sparse_seed_to_proba[2203],
    ]

    blended = np.zeros_like(ordered_components[0], dtype=np.float64)
    for model_idx, proba in enumerate(ordered_components):
        blended += proba * CLASSWISE_WEIGHTS[model_idx][None, :]

    pred_idx = (blended + OFFSETS).argmax(axis=1)
    pred_labels = [labels[idx] for idx in pred_idx]

    submission = pd.DataFrame({"ID": test_df["ID"], "target": pred_labels})
    submission.to_csv("submission.csv", index=False)
    print(submission)


if __name__ == "__main__":
    main()
