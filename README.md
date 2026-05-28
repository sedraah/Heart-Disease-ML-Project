# Heart Disease Classification 

## Overview

An end-to-end binary classification project that predicts the presence of heart disease using a cleaned, pre-encoded dataset. The workflow covers exploratory data analysis, train/test splitting with leak-free scaling, and a structured comparison of six machine learning models evaluated on accuracy, recall, and F1 score.

**Data source:** [Heart Disease — Kaggle (Amir Mahdi Abbootalebi)](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease)

---

## Dataset

| Property | Detail |
|---|---|
| File | `cleaned_heart_disease.csv` |
| Rows | 303 |
| Columns | 24 (23 features + 1 target) |
| Missing values | None |
| Duplicate rows | None |
| Target column | `heart_disease` (0 = no heart disease, 1 = heart disease) |

### Preprocessing (done prior to this project)

Categorical variables were one-hot encoded before the notebook was run. The resulting binary indicator columns are:

- `chest_pain_type_*` — chest pain type categories
- `resting_ecg_*` — resting ECG result categories
- `slope_*` — slope of the peak exercise ST segment
- `thal_fixed`, `thal_normal`, `thal_reversible` — thalassemia type

### Numeric features

`age`, `sex`, `resting_bp`, `cholesterol`, `fasting_bs`, `max_heart_rate`, `exercise_angina`, `oldpeak`, `num_major_vessels`


---

## Workflow

### Part 1: Dataset Inspection
Loaded the dataset and printed shape, head, info, describe, missing value counts, duplicate row counts, and class distribution for the target variable.

### Part 2: Visual EDA, Split, and Scaling
- Scatter plot of `age` vs `max_heart_rate` colored by `heart_disease` class.
- 80/20 stratified train/test split.
- `StandardScaler` fitted **on `X_train` only**, then applied to both `X_train` and `X_test` to prevent data leakage.

### Part 3: Baseline and Logistic Regression
- `DummyClassifier(strategy="most_frequent")` as a naive baseline.
- `LogisticRegression(max_iter=1000, random_state=42)` trained on scaled data.
- Metrics reported: accuracy, confusion matrix, classification report, ROC curve, and AUC.

### Part 4: Decision Tree and Feature Importance
- `DecisionTreeClassifier(max_depth=4, random_state=42)` trained on **unscaled** data.
- Test accuracy and F1 score reported.
- Top-6 features plotted by importance using a sorted bar chart.

### Part 5: KNN, MLP, Random Forest, and Final Comparison
- `KNeighborsClassifier(n_neighbors=5)` — scaled data.
- `MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)` — scaled data.
- `RandomForestClassifier(n_estimators=200, random_state=42)` — unscaled data.

---

## Model Comparison Results

| Model | Accuracy | Recall | F1 Score |
|---|---|---|---|
| Dummy | ~0.721 | 0.000 | 0.000 |
| Logistic Regression | ~0.836 | ~0.529 | ~0.642 |
| Decision Tree | ~0.803 | ~0.352 | ~0.5 |
| KNN | ~0.836 | ~0.529 | ~0.643 |
| **MLP** | **~0.852** | **~0.588** | **~0.690** |
| Random Forest | ~0.836 | ~0.529 | ~0.642 |

> **Best model: MLP** with the highest F1 score of ~0.690.

---

## Key Findings

- The Dummy classifier achieves ~72% accuracy by always predicting the majority class, demonstrating that accuracy alone is misleading on imbalanced data; it scores zero on recall and F1 for the positive class.
- **MLP** achieved the best balance of precision and recall (F1 ≈ 0.69), outperforming all other models.
- **Scaling matters** for distance- and magnitude-sensitive models (Logistic Regression, KNN, MLP) because unscaled features with large ranges can dominate the learning signal.
- **Tree-based models** (Decision Tree, Random Forest) are scale-invariant, they split on feature thresholds independently, so scaling has no effect on their predictions.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```


## Usage

Open `HeartDisease_notebook.ipynb` in Jupyter or Google Colab and run all cells top to bottom. Ensure `cleaned_heart_disease.csv` is in the same directory (or update the file path in the loading cell).
