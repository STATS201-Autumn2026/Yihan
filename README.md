# Robust Credit Allocation under Uncertainty

## 1. Research Question

**How can machine learning and robust optimization be integrated to improve credit allocation decisions under uncertainty in loan default predictions?**

This project investigates how machine learning and robust optimization can be combined to improve credit allocation decisions in lending markets.

The main objective is to predict borrowers' probabilities of default (PD) using machine learning and investigate whether robust optimization can improve credit allocation by balancing expected returns and credit risk.

The empirical unit of observation is an individual loan.

At the current stage, the project focuses on preliminary data exploration and validation. The next stages will involve default prediction and portfolio optimization under uncertainty.

## 2. Dataset

**Dataset:** Lending Club Loan Accepted (`lending_club_loan_two`)

**Hugging Face repository:**  
https://huggingface.co/datasets/DKU-STATS201-Autumn2026/Yihan_lending-club-loan-accepted

**Local file:** `lending_club_loan_two.csv`

**License:** CC0 1.0 (as indicated by the dataset page)

### Dataset Overview

- Number of observations: 396,030
- Number of variables: 27
- Unit of observation: Individual loan
- Duplicate records: 0

The dataset contains information about borrowers, loan characteristics, credit history, and repayment outcomes.

| Category | Variables |
|---|---|
| Loan characteristics | `loan_amnt`, `term`, `int_rate`, `installment`, `purpose` |
| Borrower characteristics | `annual_inc`, `emp_length`, `home_ownership`, `verification_status` |
| Credit history | `grade`, `sub_grade`, `dti`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `mort_acc` |
| Loan outcome | `loan_status` |

### Target Variable

For the preliminary default analysis, the repayment outcome is converted into a binary variable:

- `default = 1`: Charged Off
- `default = 0`: Fully Paid

Only loans with these two resolved repayment outcomes are included in the default analysis.

## 3. Environment and Dependencies

The project uses Python and was developed in PyCharm with a project-specific virtual environment.

### Required Libraries

- pandas
- numpy
- matplotlib
- seaborn




## 4. Preliminary Data Exploration

The preliminary analysis consists of three main parts.

### 4.1 Dataset Overview and Data Quality

The analysis examines:

- Dataset dimensions and variable types
- Missing values and their percentages
- Duplicate records
- Loan status distribution
- Descriptive statistics

### 4.2 Default Analysis

The analysis:

- Identifies loans with resolved repayment outcomes.
- Constructs a binary default indicator.
- Calculates the overall default rate.
- Examines the distributions of numerical variables.
- Compares borrower and loan characteristics across default outcomes.
- Calculates default rates across credit grades.

### 4.3 Relationships between Variables

The analysis examines:

- Correlations among numerical variables.
- Default rates across different loan terms.
- Default rates across loan purposes.
- Default rates across employment lengths.

## 5. Preliminary Results

### 5.1 Dataset Quality

The dataset contains 396,030 observations and 27 variables, with no duplicate rows.

The variables with the largest proportions of missing values are:

| Variable | Missing Count | Missing Percentage |
|---|---:|---:|
| `mort_acc` | 37,795 | 9.54% |
| `emp_title` | 22,927 | 5.79% |
| `emp_length` | 18,301 | 4.62% |
| `title` | 1,756 | 0.44% |
| `pub_rec_bankruptcies` | 535 | 0.14% |
| `revol_util` | 276 | 0.07% |

Most variables have no missing values. Missing values in the remaining variables will need to be addressed during preprocessing.

### 5.2 Loan Repayment Outcomes

| Loan Status | Number of Loans | Percentage |
|---|---:|---:|
| Fully Paid | 318,357 | 80.39% |
| Charged Off | 77,673 | 19.61% |
| **Total** | **396,030** | **100.00%** |

The overall observed default rate is **19.61%**, calculated as the proportion of Charged Off loans among all observations.

### 5.3 Default Rate by Credit Grade

| Credit Grade | Number of Loans | Default Rate |
|---|---:|---:|
| A | 64,187 | 6.29% |
| B | 116,018 | 12.57% |
| C | 105,987 | 21.18% |
| D | 63,524 | 28.87% |
| E | 31,488 | 37.36% |
| F | 11,772 | 42.79% |
| G | 3,054 | 47.84% |

The observed default rate increases from Grade A to Grade G. This pattern suggests that credit grade may be an informative feature for predicting default risk.

### 5.4 Default Rate by Loan Term

| Loan Term | Number of Loans | Default Rate |
|---|---:|---:|
| 36 months | 302,005 | 15.77% |
| 60 months | 94,025 | 31.94% |

The observed default rate for 60-month loans is approximately twice that of 36-month loans. Loan term may therefore be an important candidate predictor in the subsequent machine learning model.

### 5.5 Default Rate by Loan Purpose

| Loan Purpose | Default Rate |
|---|---:|
| small_business | 29.45% |
| moving | 23.48% |
| renewable_energy | 23.40% |
| medical | 21.71% |
| other | 21.22% |
| debt_consolidation | 20.74% |
| house | 19.72% |
| vacation | 18.92% |
| home_improvement | 17.01% |
| credit_card | 16.71% |
| major_purchase | 16.47% |
| educational | 16.34% |
| car | 13.48% |
| wedding | 12.09% |

Default rates vary across loan purposes, indicating potential differences in credit risk across loan categories.

These results are descriptive and do not establish causal relationships.

## 6. Generated Outputs

All preliminary analysis results are saved in the `result/` directory.

### CSV Files

| File | Description |
|---|---|
| `01_dataset_overview.csv` | Dataset size and duplicate records |
| `02_data_types_and_missing.csv` | Variable types and missing values |
| `03_descriptive_statistics.csv` | Descriptive statistics |
| `04_missing_values.csv` | Missing-value summary |
| `05_loan_status_distribution.csv` | Loan status distribution |
| `06_default_distribution.csv` | Default outcome distribution |
| `07_default_rate_by_grade.csv` | Default rates by credit grade |
| `08_correlation_matrix.csv` | Numerical-variable correlations |
| `09_default_rate_by_term.csv` | Default rates by loan term |
| `10_default_rate_by_purpose.csv` | Default rates by loan purpose |
| `11_default_rate_by_employment_length.csv` | Default rates by employment length |

### Figures

| File | Description |
|---|---|
| `01_missing_values.png` | Missing values by variable |
| `02_loan_status_distribution.png` | Distribution of loan statuses |
| `03_numerical_distributions.png` | Distributions of numerical variables |
| `04_features_by_default_status.png` | Numerical features by default status |
| `05_default_rate_by_grade.png` | Default rates across credit grades |
| `06_correlation_matrix.png` | Correlation heatmap |
| `07_default_rate_by_term.png` | Default rates by loan term |
| `08_default_rate_by_purpose.png` | Default rates by loan purpose |

## 7. Next Steps

The preliminary results provide a foundation for the next stages of the project.

The planned research process is:

1. **Data preprocessing:** Handle missing values, encode categorical variables, and prepare model features.
2. **Default prediction:** Train machine learning models to estimate borrowers' probabilities of default.
3. **Model evaluation:** Evaluate predictive performance using ROC-AUC, Brier score, and probability calibration.
4. **Portfolio optimization:** Incorporate predicted default probabilities and expected loan returns into a credit allocation framework.
5. **Robust optimization:** Account for uncertainty in predicted default probabilities rather than relying only on point estimates.
6. **Comparison and sensitivity analysis:** Compare conventional optimization with robust optimization in terms of expected returns, credit risk, and allocation decisions.

A key research objective is to investigate whether accounting for uncertainty in default predictions can lead to more reliable credit allocation decisions.

