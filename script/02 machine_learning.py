# ============================================================
# 02. Machine Learning: Loan Default Prediction
# Goal: Predict the probability of loan default (PD)
# Models: Logistic Regression and Random Forest
# Evaluation: ROC-AUC, PR-AUC, Brier Score, Accuracy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.calibration import calibration_curve


# ============================================================
# 1. Set paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "lending_club_loan_two.csv"

RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 3. Create default target variable
# ============================================================

# Charged Off = 1 (default)
# Fully Paid = 0 (non-default)

df["default"] = df["loan_status"].map({
    "Charged Off": 1,
    "Fully Paid": 0
})

# Keep only observations with known outcomes
df = df.dropna(subset=["default"])

df["default"] = df["default"].astype(int)

print("\nDefault distribution:")
print(df["default"].value_counts())

print("\nDefault rate:")
print(df["default"].mean())


# ============================================================
# 4. Select features
# ============================================================

# Exclude:
# loan_status: target variable
# default: target variable
# address: personal/location information
# title: high-cardinality text field
# emp_title: high-cardinality text field
# issue_d: excluded from this initial baseline model
# earliest_cr_line: date field, handled separately in a later version

drop_columns = [
    "loan_status",
    "default",
    "address",
    "title",
    "emp_title",
    "issue_d",
    "earliest_cr_line"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["default"]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])


# ============================================================
# 5. Identify numerical and categorical variables
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 6. Split training and testing data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)


# ============================================================
# 7. Preprocessing
# ============================================================

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 8. Define machine learning models
# ============================================================

models = {

    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

}


# ============================================================
# 9. Train models and evaluate performance
# ============================================================

results = []

predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict default probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Predict binary outcomes using 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)

    predictions[model_name] = y_prob

    # Evaluation metrics
    roc_auc = roc_auc_score(y_test, y_prob)

    pr_auc = average_precision_score(y_test, y_prob)

    brier = brier_score_loss(y_test, y_prob)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": model_name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Brier_Score": brier,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })

    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fully Paid", "Charged Off"],
        yticklabels=["Fully Paid", "Charged Off"]
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        RESULT_DIR / f"02_confusion_matrix_{filename}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. Save model comparison
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "12_model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# 11. Plot ROC curves
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, y_prob in predictions.items():

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "13_roc_curve_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. Plot calibration curves
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, y_prob in predictions.items():

    prob_true, prob_pred = calibration_curve(
        y_test,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label=model_name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Default Probability")
plt.ylabel("Observed Default Rate")
plt.title("Calibration Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "14_calibration_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. Save test-set predicted default probabilities
# ============================================================

prediction_df = pd.DataFrame({
    "actual_default": y_test.values,
    "logistic_regression_PD": predictions["Logistic Regression"],
    "random_forest_PD": predictions["Random Forest"]
})

prediction_df.to_csv(
    RESULT_DIR / "15_test_set_default_predictions.csv",
    index=False
)


# ============================================================
# 14. Save a copy of the trained models
# ============================================================

import joblib

joblib.dump(
    models["Logistic Regression"],
    RESULT_DIR / "logistic_regression_model.pkl"
)

joblib.dump(
    models["Random Forest"],
    RESULT_DIR / "random_forest_model.pkl"
)


# ============================================================
# 15. Finish
# ============================================================

print("\n" + "=" * 60)
print("Machine Learning Analysis Completed!")
print("=" * 60)

print("All results saved in:")
print(RESULT_DIR)