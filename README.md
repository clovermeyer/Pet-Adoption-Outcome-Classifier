# Pet Adoption Outcome Classifier

A machine learning project that predicts whether a shelter animal will be adopted based on biological and environmental attributes.

**Original work by Clover Meyer** — extended and restructured for GitHub from Google Colab.

---

## Project Overview

Using a dataset of 200 shelter animal records, this project trains and compares four classification models to predict adoption outcomes (`True` / `False`).

| Feature | Description |
|---|---|
| `age_years` | Age of the animal in years |
| `species` | Cat, Dog, Bird, Hamster, or Rabbit |
| `gender` | Male / Female |
| `color` | Primary coat color |
| `breed` | Breed string |

**Target variable:** `adopted` (boolean)

---

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| k-Nearest Neighbors (k=5) | Distance-based, non-parametric |
| Decision Tree | Fully interpretable, visualisable |
| **Random Forest** | **Best performer; provides feature importances** |

> **Why Decision Tree & Random Forest were added:** The original notebook's conclusion explicitly recommended tree-based models as the logical next step, noting that Logistic Regression and KNN lacked the complexity to capture non-linear relationships in the data. This version implements both.

---

## Key Outputs

| File | Description |
|---|---|
| `outputs/eda_charts.png` | Adoption % by species and gender |
| `outputs/cm_*.png` | Confusion matrix for each model |
| `outputs/model_comparison.png` | CV vs hold-out accuracy comparison chart |
| `outputs/feature_importance.png` | Top 20 features by Random Forest importance |
| `outputs/decision_tree.png` | Visualised Decision Tree (first 3 levels) |
| `models/best_model_*.pkl` | Serialised best-performing model (joblib) |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/pet-adoption-classifier.git
cd pet-adoption-classifier

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. Place your `adoption1.csv` file in the `data/` folder.
2. Run the classifier:

```bash
python pet_adoption_classifier.py
```

All charts and the best model will be saved automatically.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## Results Summary

> Results will vary slightly based on random state, but expected ranges:

| Model | CV Accuracy |
|---|---|
| Logistic Regression | ~52% |
| k-Nearest Neighbors | ~45% |
| Decision Tree | ~55–60% |
| Random Forest | ~58–65% |

**Why are all accuracies modest?** As discussed in the original notebook, the selected features (age, species, gender, color, breed) appear to have limited predictive power for adoption outcomes. Untracked variables — pet personality, health status, adopter history, time in shelter — likely explain far more variance. This is an honest and expected result; the project correctly identifies it as a data limitation, not a code failure.

---

## Suggested Next Steps

- Collect richer features (personality scores, health status, days in shelter)
- Try Gradient Boosting (XGBoost / LightGBM)
- Hyperparameter tune Random Forest via `GridSearchCV`
- Expand to a larger dataset

---

## File Structure

```
pet-adoption-classifier/
├── data/
│   └── adoption1.csv          
├── models/
│   └── best_model_*.pkl       
├── outputs/
│   ├── eda_charts.png
│   ├── cm_*.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── decision_tree.png
├── pet_adoption_classifier.py
├── requirements.txt
└── README.md
```
