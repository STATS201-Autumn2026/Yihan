
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# 0. SETUP
# ============================================================

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Input dataset
DATA_PATH = BASE_DIR / "lending_club_loan_two.csv"

# Output directory
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Results will be saved to:", RESULT_DIR)


# Helper function: save figures
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 1. DATASET OVERVIEW AND DATA QUALITY
# ============================================================

print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

# Basic information
overview = pd.DataFrame({
    "Metric": [
        "Number of observations",
        "Number of variables",
        "Duplicate rows"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        df.duplicated().sum()
    ]
})

print(overview.to_string(index=False))
overview.to_csv(RESULT_DIR / "01_dataset_overview.csv", index=False)

# Data types
data_types = pd.DataFrame({
    "Variable": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Count": df.isnull().sum().values,
    "Missing Percentage (%)": (df.isnull().mean() * 100).round(2).values
})

print("\nData types and missing values:")
print(data_types.to_string(index=False))

data_types.to_csv(RESULT_DIR / "02_data_types_and_missing.csv", index=False)

# Descriptive statistics
descriptive_stats = df.describe(include="all").T
descriptive_stats.to_csv(RESULT_DIR / "03_descriptive_statistics.csv")

# Missing values
missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage (%)": df.isnull().mean() * 100
})

missing = missing.sort_values(
    "Missing Percentage (%)",
    ascending=False
)

missing.to_csv(RESULT_DIR / "04_missing_values.csv")

# Missing value plot
missing_plot = missing[missing["Missing Count"] > 0]

if len(missing_plot) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=missing_plot.index,
        y=missing_plot["Missing Percentage (%)"]
    )
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Missing Values (%)")
    plt.title("Missing Values by Variable")
    save_plot("01_missing_values.png")

# Loan status distribution
loan_status = (
    df["loan_status"]
    .value_counts(dropna=False)
    .rename_axis("Loan Status")
    .reset_index(name="Count")
)

loan_status["Percentage (%)"] = (
    loan_status["Count"] / len(df) * 100
)

print("\nLoan status distribution:")
print(loan_status.to_string(index=False))

loan_status.to_csv(
    RESULT_DIR / "05_loan_status_distribution.csv",
    index=False
)

plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x="loan_status",
    order=df["loan_status"].value_counts().index
)
plt.title("Distribution of Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Loans")
plt.xticks(rotation=35, ha="right")
save_plot("02_loan_status_distribution.png")


# ============================================================
# 2. DEFAULT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("2. DEFAULT ANALYSIS")
print("=" * 60)

# Keep loans with resolved repayment outcomes
valid_status = ["Fully Paid", "Charged Off"]

eda_df = df[df["loan_status"].isin(valid_status)].copy()

# Define default indicator:
# 1 = Charged Off
# 0 = Fully Paid
eda_df["default"] = (
    eda_df["loan_status"] == "Charged Off"
).astype(int)

print("Original observations:", len(df))
print("Resolved observations:", len(eda_df))
print("Excluded observations:", len(df) - len(eda_df))

# Default distribution
default_distribution = (
    eda_df["default"]
    .value_counts()
    .rename(index={0: "Fully Paid", 1: "Charged Off"})
    .rename_axis("Outcome")
    .reset_index(name="Count")
)

default_distribution["Percentage (%)"] = (
    default_distribution["Count"] / len(eda_df) * 100
)

print("\nDefault distribution:")
print(default_distribution.to_string(index=False))

print(
    f"\nOverall default rate: "
    f"{eda_df['default'].mean() * 100:.2f}%"
)

default_distribution.to_csv(
    RESULT_DIR / "06_default_distribution.csv",
    index=False
)

# Numerical variables
numeric_vars = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti"
]

numeric_vars = [
    col for col in numeric_vars
    if col in eda_df.columns
]

# Histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(numeric_vars):
    sns.histplot(
        data=eda_df,
        x=col,
        bins=40,
        kde=True,
        ax=axes[i]
    )
    axes[i].set_title(f"Distribution of {col}")

for i in range(len(numeric_vars), len(axes)):
    axes[i].axis("off")

save_plot("03_numerical_distributions.png")

# Boxplots by default status
comparison_vars = [
    "int_rate",
    "loan_amnt",
    "annual_inc",
    "dti"
]

comparison_vars = [
    col for col in comparison_vars
    if col in eda_df.columns
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, col in zip(axes, comparison_vars):
    sns.boxplot(
        data=eda_df,
        x="default",
        y=col,
        ax=ax
    )
    ax.set_title(f"{col} by Default Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fully Paid", "Charged Off"])

for i in range(len(comparison_vars), len(axes)):
    axes[i].axis("off")

save_plot("04_features_by_default_status.png")

# Default rate by credit grade
grade_default = (
    eda_df.groupby("grade", observed=True)["default"]
    .agg(["mean", "count"])
    .reset_index()
)

grade_default["default_rate (%)"] = grade_default["mean"] * 100

grade_default = grade_default.sort_values("grade")

print("\nDefault rate by credit grade:")
print(grade_default.to_string(index=False))

grade_default.to_csv(
    RESULT_DIR / "07_default_rate_by_grade.csv",
    index=False
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=grade_default,
    x="grade",
    y="default_rate (%)",
    order=grade_default["grade"].dropna().tolist()
)
plt.title("Default Rate by Credit Grade")
plt.xlabel("Credit Grade")
plt.ylabel("Default Rate (%)")
save_plot("05_default_rate_by_grade.png")


# ============================================================
# 3. RELATIONSHIPS BETWEEN VARIABLES
# ============================================================

print("\n" + "=" * 60)
print("3. RELATIONSHIPS BETWEEN VARIABLES")
print("=" * 60)

# Correlation matrix
correlation_vars = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "default"
]

correlation_vars = [
    col for col in correlation_vars
    if col in eda_df.columns
]

corr = eda_df[correlation_vars].corr()

corr.to_csv(RESULT_DIR / "08_correlation_matrix.csv")

plt.figure(figsize=(13, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("Correlation Matrix")
save_plot("06_correlation_matrix.png")


# Default rate by loan term
if "term" in eda_df.columns:

    term_default = (
        eda_df.groupby("term", observed=True)["default"]
        .agg(["mean", "count"])
        .reset_index()
    )

    term_default["default_rate (%)"] = term_default["mean"] * 100

    print("\nDefault rate by loan term:")
    print(term_default.to_string(index=False))

    term_default.to_csv(
        RESULT_DIR / "09_default_rate_by_term.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=term_default,
        x="term",
        y="default_rate (%)"
    )
    plt.title("Default Rate by Loan Term")
    plt.xlabel("Loan Term")
    plt.ylabel("Default Rate (%)")
    save_plot("07_default_rate_by_term.png")


# Default rate by loan purpose
if "purpose" in eda_df.columns:

    purpose_default = (
        eda_df.groupby("purpose", observed=True)["default"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nDefault rate by loan purpose:")
    print(purpose_default.to_string())

    purpose_default.to_csv(
        RESULT_DIR / "10_default_rate_by_purpose.csv"
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x=purpose_default.values,
        y=purpose_default.index
    )
    plt.title("Default Rate by Loan Purpose")
    plt.xlabel("Default Rate (%)")
    plt.ylabel("Loan Purpose")
    save_plot("08_default_rate_by_purpose.png")


# Default rate by employment length
if "emp_length" in eda_df.columns:

    emp_default = (
        eda_df.groupby("emp_length", observed=True)["default"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    emp_default.to_csv(
        RESULT_DIR / "11_default_rate_by_employment_length.csv"
    )


# ============================================================
# 4. FINISH
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"All results have been saved to:\n{RESULT_DIR}")

print("\nFiles generated:")
for file in sorted(RESULT_DIR.iterdir()):
    print(file.name)
