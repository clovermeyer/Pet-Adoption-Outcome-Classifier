"""
Pet Adoption Outcome Classifier
================================
Predicts whether a shelter animal will be adopted (True/False) based on
biological and environmental attributes.

Dataset features:
    age_years  : pet age (numerical, -1 encoded as missing)
    species    : Cat, Dog, Bird, Hamster, Rabbit
    gender     : Male / Female
    color      : primary coat color
    breed      : breed string

Target:
    adopted    : bool — True if the pet was adopted

Models compared:
    1. Logistic Regression   (baseline, linear boundary)
    2. k-Nearest Neighbors   (k=5, distance-based)
    3. Decision Tree         (non-linear, interpretable)  ← NEW
    4. Random Forest         (ensemble, best expected)    ← NEW

New additions vs. original notebook:
    - Decision Tree & Random Forest (recommended in original conclusion, never built)
    - 5-fold cross-validation on all models (single 80/20 split unreliable on n=200)
    - Feature importance visualization (Random Forest)
    - Consolidated model comparison bar chart
    - Modular, reusable functions
    - Persisted best model to disk via joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics           import (accuracy_score, confusion_matrix,
                                       ConfusionMatrixDisplay, classification_report)
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.tree              import DecisionTreeClassifier, plot_tree
import joblib

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & INSPECT
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and print basic diagnostics."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing / sentinel values (-1) per column:")
    print((df == -1).sum())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRELIMINARY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame) -> None:
    """Generate and save exploratory bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Adoption % by species
    species_pct = (df["species"].value_counts() / len(df) * 100).reset_index()
    species_pct.columns = ["species", "percentage"]
    sns.barplot(ax=axes[0], x="species", y="percentage",
                data=species_pct, palette="viridis", hue="species", legend=False)
    axes[0].set_title("Adoption Percentage by Species")
    axes[0].set_ylabel("Adoption Percentage (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Adoption rate by gender
    gender_rate = df.groupby("gender")["adopted"].mean() * 100
    gender_rate = gender_rate.reset_index()
    gender_rate.columns = ["gender", "adoption_rate"]
    sns.barplot(ax=axes[1], x="gender", y="adoption_rate",
                data=gender_rate, palette="magma", hue="gender", legend=False)
    axes[1].set_title("Adoption Rate by Gender")
    axes[1].set_ylabel("Adoption Rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("outputs/eda_charts.png", dpi=150)
    plt.show()
    print("Saved → outputs/eda_charts.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Steps:
      1. Replace sentinel -1 with NaN in numerical columns
      2. Select features & target
      3. Mean-impute age_years
      4. One-hot encode categoricals (drop_first to avoid multicollinearity)
      5. Train/test split (80/20, stratified to preserve class balance)
      6. StandardScaler on age_years only

    WHY STRATIFIED SPLIT:
        With only 200 records a random split can produce imbalanced folds.
        stratify=y ensures both splits mirror the overall adoption rate.
    """
    # 1. Sentinel → NaN
    cleaned = df.copy()
    for col in ["age_years", "adopter_age"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(-1, np.nan)

    # 2. Features & target
    features = ["age_years", "species", "gender", "color", "breed"]
    X = cleaned[features].copy()
    y = cleaned["adopted"]

    # 3. Impute age_years mean
    X["age_years"] = X["age_years"].fillna(X["age_years"].mean())

    # 4. One-hot encode
    X = pd.get_dummies(X, columns=["species", "gender", "color", "breed"],
                       drop_first=True)

    print(f"\nFeature shape after encoding : {X.shape}")
    print(f"Class distribution (adopted) :\n{y.value_counts()}")

    # 5. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Scale age_years (important for KNN & Logistic Regression)
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train["age_years"] = scaler.fit_transform(X_train[["age_years"]])
    X_test["age_years"]  = scaler.transform(X_test[["age_years"]])

    print(f"\nTrain size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Return dict of model_name → estimator.

    WHY DECISION TREE:
        The original notebook conclusion explicitly recommended tree-based
        models as a next step.  A single Decision Tree is fully interpretable —
        you can visualise exactly which feature the model splits on at every
        node, making it useful for explaining predictions.

    WHY RANDOM FOREST:
        An ensemble of 100 decision trees that reduces overfitting through
        bagging.  It also provides feature_importances_ — a ranked list of
        which attributes most influence adoption outcomes — which is the key
        analytic insight missing from the original notebook.
    """
    return {
        "Logistic Regression" : LogisticRegression(random_state=42, solver="liblinear", max_iter=1000),
        "k-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
        "Decision Tree"       : DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION + HOLD-OUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(models: dict, X_train, X_test, y_train, y_test, X_full, y_full) -> pd.DataFrame:
    """
    WHY CROSS-VALIDATION:
        With n=200, a single 80/20 split produces a test set of only 40 samples.
        A lucky/unlucky split can swing accuracy by ±10 pp.
        5-fold stratified CV gives five independent evaluations and reports the
        mean ± std, which is a far more honest estimate of generalisation.
    """
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        # Cross-val on full dataset
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="accuracy")

        # Hold-out evaluation
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        holdout  = accuracy_score(y_test, y_pred)

        results.append({
            "Model"          : name,
            "CV Mean Acc"    : cv_scores.mean(),
            "CV Std"         : cv_scores.std(),
            "Hold-out Acc"   : holdout,
        })

        print(f"\n{'─'*40}")
        print(f"  {name}")
        print(f"  CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Hold-out acc  : {holdout:.4f}")
        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        cm      = confusion_matrix(y_test, y_pred)
        cmap    = {"Logistic Regression": plt.cm.Blues,
                   "k-Nearest Neighbors": plt.cm.Oranges,
                   "Decision Tree"      : plt.cm.Purples,
                   "Random Forest"      : plt.cm.Greens}.get(name, plt.cm.Blues)
        ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap=cmap, ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(f"outputs/cm_{safe_name}.png", dpi=150)
        plt.show()

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results_df: pd.DataFrame) -> None:
    """Side-by-side bar chart: CV accuracy vs hold-out accuracy for all models."""
    x     = np.arange(len(results_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, results_df["CV Mean Acc"],  width, label="CV Mean Accuracy",
                   color="steelblue",   yerr=results_df["CV Std"], capsize=4)
    bars2 = ax.bar(x + width/2, results_df["Hold-out Acc"], width, label="Hold-out Accuracy",
                   color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=10, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — CV vs Hold-out Accuracy\n(error bars = ±1 std over 5 folds)")
    ax.legend()
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.show()
    print("Saved → outputs/model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(rf_model, feature_names: list, top_n: int = 20) -> None:
    """
    WHY THIS MATTERS:
        The original notebook never answered *which* features actually drive
        adoption.  The Random Forest's feature_importances_ attribute gives a
        ranked answer.  This is the core analytic insight of the whole project.
    """
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top         = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.show()
    print("Saved → outputs/feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. DECISION TREE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_tree(dt_model, feature_names: list) -> None:
    """
    WHY THIS IS VALUABLE:
        Unlike the other models, a Decision Tree's logic can be read like a
        flowchart.  Visualising the first 3 levels shows the most impactful
        splitting decisions in plain English — useful for explaining to a
        non-technical audience (shelter staff, report readers, etc.).
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(dt_model, feature_names=feature_names,
              class_names=["Not Adopted", "Adopted"],
              filled=True, max_depth=3, ax=ax, fontsize=8,
              impurity=False, proportion=True)
    ax.set_title("Decision Tree (first 3 levels) — Pet Adoption Classifier")
    plt.tight_layout()
    plt.savefig("outputs/decision_tree.png", dpi=120)
    plt.show()
    print("Saved → outputs/decision_tree.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PERSIST BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

def save_best_model(models: dict, results_df: pd.DataFrame, X_train, y_train) -> None:
    """Save the best-performing model (by CV mean accuracy) using joblib."""
    best_name  = results_df.loc[results_df["CV Mean Acc"].idxmax(), "Model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    path = f"models/best_model_{best_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(best_model, path)
    print(f"\nBest model : {best_name}  (CV acc = {results_df['CV Mean Acc'].max():.4f})")
    print(f"Saved      → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Change this path to match your local CSV location ──
    DATA_PATH = "data/adoption1.csv"

    df = load_data(DATA_PATH)

    plot_eda(df)

    X_train, X_test, y_train, y_test, X_full, y_full = preprocess(df)

    models = build_models()

    results_df = evaluate_models(models, X_train, X_test, y_train, y_test, X_full, y_full)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    plot_comparison(results_df)

    # Random Forest feature importance
    rf = models["Random Forest"]
    rf.fit(X_train, y_train)
    plot_feature_importance(rf, list(X_train.columns))

    # Decision Tree visualisation
    dt = models["Decision Tree"]
    dt.fit(X_train, y_train)
    plot_decision_tree(dt, list(X_train.columns))

    save_best_model(models, results_df, X_train, y_train)

    print("\nAll outputs saved to outputs/")
    print("Best model saved to models/")
