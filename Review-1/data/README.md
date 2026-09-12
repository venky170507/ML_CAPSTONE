# Adult Income Dataset

This directory should contain:

```text
AdultIncome.csv
```

The notebook/helper automatically downloads the **Adult Income** dataset through
scikit-learn's OpenML interface when the CSV is missing.

Dataset source:

- UCI Machine Learning Repository — Adult dataset
- OpenML representation of the UCI Adult dataset

Target:

```text
income
```

Classes:

```text
<=50K
>50K
```

Do not manually preprocess the CSV before running the notebooks. The notebooks
perform the required missing-value handling, categorical encoding, scaling, and
train/test split inside a leakage-safe scikit-learn Pipeline.
