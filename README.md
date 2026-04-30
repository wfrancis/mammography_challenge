# Mammography Report Classification

Code for the [SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification) Kaggle competition: predict the BI-RADS Category (0–6) from the text of a Portuguese-language mammogram radiology report. Evaluation metric: macro F1.

## Approach (best submission)

A **multi-head TF-IDF + LightGBM ensemble** with per-class threshold tuning, a specialized binary class-0 detector, and a small set of clinical-keyword guardrails. CPU-only, ~3m 40s end-to-end.

See [`spr-2026-mammo-multihead-thresh-tuned/spr-2026-mammo-multihead-thresh-tuned.py`](spr-2026-mammo-multihead-thresh-tuned/spr-2026-mammo-multihead-thresh-tuned.py).

### Pipeline

1. **Two text representations** of each report:
   - `achados`: the clinical-findings section, isolated by regex between `achados:` and `análise comparativa:` markers
   - `full`: the full report with whitespace normalization
   - All numeric tokens masked to `NUM` to prevent the model from latching onto specific measurements

2. **Three TF-IDF heads:**

   | Head | Source | Word n-grams | Char n-grams | Max char features |
   |------|--------|--------------|--------------|-------------------|
   | A    | achados| 1–3          | char_wb 3–5  | 80,000            |
   | F    | full   | 1–3          | char_wb 3–5  | 80,000            |
   | F2   | full   | 1–3          | char_wb 3–6  | 100,000           |

   All vectorizers use `sublinear_tf=True`, `min_df=3`, `max_df=0.95`.

3. **Dense clinical features** (computed in addition to TF-IDF):
   - `report_length`
   - `has_measurement` — regex `\b(cm|mm|medindo)\b`
   - `has_spiculation` — regex `espiculad`
   - `has_distortion` — regex `distorção arquitetural`
   - `has_biopsy` — regex `biopsy|biópsia|resultado de cine|carcinoma`

4. **Models:**
   - Three calibrated `LinearSVC`s (one per TF-IDF head), each `class_weight='balanced'`, `cv=3`, sigmoid calibration
   - One LightGBM classifier on (Head A TF-IDF + dense features), 300 trees, `learning_rate=0.05`, `max_depth=6`, `class_weight='balanced'`

5. **Ensemble:**
   ```
   SVC_ensemble = 0.25·P_A + 0.40·P_F + 0.35·P_F2
   final_proba  = 0.70·SVC_ensemble + 0.30·P_LGB
   ```

6. **Per-class probability thresholds** (grid-searched on out-of-fold predictions, +0.01867 macro F1 over uniform thresholds):
   ```
   class 6 if P[6] > 0.10
   class 5 if P[5] > 0.15 and not class 6
   class 4 if P[4] > 0.23 and not class 5/6
   class 3 if P[3] > 0.38 and not class 4/5/6
   else argmax(P)
   ```

7. **Binary class-0 detector** — a separate calibrated `LinearSVC` trained on the `{0, 2}` subset using Head F features. At inference, if the main ensemble predicts class 2 but `P(class 0) > 0.55`, override to class 0. Class-0-vs-2 was the dominant confusion pair on OOF.

8. **Clinical guardrails** (regex post-processing):
   - `resultado de cine grau 3 | carcinoma | \bcdis\b` → force class 6
   - `espiculad` AND `distorção` AND current prediction < 4 → force class 5

## Repository layout

```
.
├── README.md
├── LICENSE
├── NOTICE
└── spr-2026-mammo-*/                    # one folder per submission iteration
    └── spr-2026-mammo-*.py              # the submission script
```

The 11 directories contain the iteration history. The best is `spr-2026-mammo-multihead-thresh-tuned/`.

## Notable observations

- **Lightweight beat heavyweight.** Earlier transformer-based attempts (BERTimbau, etc.) underperformed this TF-IDF pipeline. The signal lives heavily in surface lexical patterns (clinical keywords, structural phrasing) that classical NLP captures well, while transformers added variance without improving generalization on the relatively small dataset.
- **Per-class threshold tuning was the single highest-leverage post-ensemble trick** (+0.01867 OOF macro F1).
- **The findings-section extraction (Head A)** stabilizes the ensemble. Full-report heads pick up boilerplate template language; restricting one head to just the clinical findings section gives the ensemble a less template-biased view.
- **Public/private gap.** The threshold-tuned ensemble had a meaningful negative gap (private > public), suggesting the test distribution rewarded calibration over leaderboard overfitting.

## Running

```bash
pip install scikit-learn scipy lightgbm pandas numpy

# Place competition data at ./data/raw/{train,test}.csv,
# OR /kaggle/input/spr-2026-mammography-report-classification/

python spr-2026-mammo-multihead-thresh-tuned/spr-2026-mammo-multihead-thresh-tuned.py
```

The script writes `submission.csv` to the working directory.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution and
modification notes.

## Citation (dataset / competition)

```
Eduardo Farina and Felipe Kitamura, MD, PhD.
SPR 2026 Mammography Report Classification.
https://kaggle.com/competitions/spr-2026-mammography-report-classification
```
