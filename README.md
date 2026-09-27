# Data Science & AI/ML Practical Exam - Set A

**Student name:** Krisha Patel

**Student ID:** 11633

**Set:** A

**Objective:** Predict campaign responses (`response`: 1 = responded, 0 = did not respond) and identify
audience segments from a synthetic marketing dataset.

## Data Dictionary

| Column | Meaning |
|---|---|
| record_id | Row identifier, excluded from every model |
| visits | Numeric predictor (synthetic index) |
| recency | Numeric predictor (synthetic index) |
| engagement | Numeric predictor (synthetic index) |
| spend | Numeric predictor (synthetic index) |
| group | Categorical operational cohort, G1 or G2 |
| response | Target: 1 = responded, 0 = did not respond |

## Generator

The raw data is created by the supplied, unchanged script `src/generate_data.py`, run once from the
repository root:

```
python src/generate_data.py
```

This creates `data/raw/set_d.csv` with 305 rows, including 5 exact duplicate rows.

## Cleaning and Split

- Exact duplicate rows are removed, leaving 300 unique records.
- 80/20 stratified train/test split, `random_state=42` -> fit_full (240) / test (60).
- 20% of fit_full is reserved for ANN validation, stratified, `random_state=42` -> fit (192) / validation (48).
- Record IDs for all three partitions are saved in `outputs/splits.csv` and confirmed disjoint.

Partition sizes: **fit = 192, validation = 48, test = 60**.

## Feature Formula

`engineered_feature = engagement / (recency + 1)`

Created after numeric imputation, using original-scale values, before scaling. The target is never used
to build this feature.

## Tools / Package Versions

See `requirements.txt`. Main libraries: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0,
matplotlib 3.10.8, tensorflow-cpu 2.21.0.

## Folder Map

```
README.md
requirements.txt
.gitignore
src/generate_data.py          supplied generator, unchanged
data/raw/set_d.csv            raw generated data (duplicates and missing values kept)
notebooks/exam.ipynb          executed notebook, five labeled sections
outputs/splits.csv            record_id and fit/validation/test membership
outputs/*.csv, outputs/*.txt  statistics, audits, metrics, predictions, cluster profiles
outputs/figures/              histogram, confusion matrix, ANN loss curve
models/                       saved ANN model and fitted preprocessing
```

## Setup and Run (from repository root)

```
pip install -r requirements.txt
python src/generate_data.py
jupyter notebook notebooks/exam.ipynb
```

Run all cells from a clean kernel. The notebook uses relative paths (`../data`, `../outputs`, `../models`)
because it is executed from inside the `notebooks/` folder.

## Statistical Hypotheses and Assumptions (M2)

- H0: mean engagement of G1 equals mean engagement of G2.
- H1: mean engagement of G1 is not equal to mean engagement of G2.
- Two-sided Welch t-test, alpha = 0.05, on the 192 fit records, observed (non-imputed) engagement values.
- G1 count = 84, G2 count = 108, t-statistic = 0.723, p-value = 0.471 -> fail to reject H0 at alpha = 0.05.
- 95% confidence interval for the overall observed mean engagement: (49.537, 52.466).
- Assumptions: independence of records, and approximate normality of the sampling distribution of the
  mean (reasonable here given the sample size).
- No causal claim is made from this result; group was not randomly assigned in an experiment.

## Key Results

**Supervised learning (test set, threshold = 0.5):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (majority class) | 0.583 | - | - | - |
| Logistic Regression | 0.817 | 0.833 | 0.857 | 0.845 |
| ANN | 0.833 | 0.879 | 0.829 | 0.853 |

Both models clearly beat the majority-class baseline on this holdout split. The ANN and logistic
regression reach a similar F1 score, with the ANN slightly ahead on this run; the ANN was not required
to outperform logistic regression.

**Clustering (K-Means, fit partition only):** k=2 gave the highest silhouette score (0.226) among
k=2,3,4, so k=2 was selected. Cluster profiles:
- Cluster with higher engagement and lower recency (n=83): named "Active Engaged Visitors".
  Practical action: prioritise this group for the campaign.
- Cluster with lower engagement and higher recency (n=109): named "Disengaged Lapsed Visitors".
  Practical action: send a re-engagement message before including them in the campaign.

Cluster IDs are arbitrary and were produced without using the response target.

## Two Numerical Findings

1. Logistic regression reaches an F1 score of 0.845 on the test set, well above the majority-class
   baseline accuracy of 0.583, so the model adds real predictive value on this data.
2. The two K-Means segments differ mainly in engagement (46.1 vs 57.4) and recency (55.2 vs 44.2),
   which suggests engagement and recency are the main drivers separating the two audience groups.

## Recommendation and Limitation

**Recommendation:** use the logistic regression or ANN model to prioritise likely responders for the
campaign, and target the "Disengaged Lapsed Visitors" segment with a re-engagement step first.

**Limitation:** all results come from one small (60-record) synthetic test split. Metrics may change with
a different split or a larger sample, so these results should not be read as a guarantee of real-world
performance.

## Prediction Reconciliation

`outputs/test_predictions_logreg.csv` and `outputs/test_predictions_ann.csv` both use the same 60 test
`record_id` values, in the same order, with the same true labels. The accuracy/precision/recall/F1
values in `outputs/model_comparison.csv` are computed directly from these saved prediction files.

## Video

- URL: https://drive.google.com/file/d/17M4AzOo8njgO2wOjLLZL2MZUfu0fVcsa/view?usp=sharing
- Duration: 15:40 

## References

- scikit-learn documentation (https://scikit-learn.org)
- TensorFlow/Keras documentation (https://www.tensorflow.org)
- SciPy documentation (https://scipy.org)

## Declaration

All work is my own except where cited.
