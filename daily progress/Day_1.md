# Customer Lifetime Value & Churn Intelligence System

## Day 1 --- Project Setup, Data Preparation & Exploratory Data Analysis

**Date:** 13 November 2026\
**Dataset:** IBM Telco Customer Churn\
**Environment:** Google Colab and Google Drive

------------------------------------------------------------------------

## 1. Project Objective

Build an end-to-end customer intelligence system to predict churn,
explain churn drivers, estimate customer survival, identify customers
who may respond to retention offers, estimate customer lifetime value
(CLV), and support business decisions through a Streamlit dashboard and
ROI simulator.

## 2. Environment and Project Setup

-   Created the project workspace in Google Drive.
-   Connected Google Colab to Google Drive.
-   Created the project folders: `data`, `notebooks`, `models`,
    `dashboard`, and `documentation`.
-   Installed and imported the initial Python libraries.

## 3. Dataset Collection

Loaded the IBM Telco Customer Churn dataset.

  Dataset detail         Result
  -------------------- --------
  Customer records        7,043
  Total columns              21
  Retained customers      5,174
  Churned customers       1,869
  Retention rate         73.46%
  Churn rate             26.54%

Saved the raw dataset in the project data folder.

## 4. Data Quality Assessment

-   Found that `TotalCharges` was stored as an object instead of a
    numeric datatype.
-   Identified 11 blank values in `TotalCharges`.
-   Investigated the affected records: they had zero tenure and no
    accumulated charges.
-   Converted `TotalCharges` to numeric and replaced those 11 blank
    values with `0`.
-   Checked for duplicate rows and duplicate customer IDs.

  Validation check                     Result
  ------------------------------- -----------
  Missing values after cleaning             0
  Duplicate rows                            0
  Duplicate customer IDs                    0
  `TotalCharges` datatype           `float64`

Saved the processed dataset as `telco_churn_processed.csv`.

## 5. Exploratory Data Analysis

Analysed customer demographics, churn distribution, tenure, contract
type, monthly charges, payment methods, internet services, and
additional services.

### 5.1 Contract Analysis

Approximate churn rates read from the chart:

  Contract type      Approximate churn rate
  ---------------- ------------------------
  Month-to-month                        43%
  One year                              11%
  Two year                               3%

**Finding:** Month-to-month customers had substantially higher churn
than customers with longer-term contracts.

### 5.2 Monthly Charges Analysis

  Churn status     Mean monthly charges   Median monthly charges
  -------------- ---------------------- ------------------------
  Retained                        61.27                    64.43
  Churned                         74.44                    79.65

**Finding:** Churned customers had higher average and median monthly
charges in this dataset.

### 5.3 Payment Method Analysis

Approximate churn rates read from the chart:

  Payment method                Approximate churn rate
  --------------------------- ------------------------
  Electronic Check                                 45%
  Mailed Check                                     19%
  Bank Transfer (automatic)                        17%
  Credit Card (automatic)                          15%

**Finding:** Electronic Check users had the highest churn rate among the
payment method categories.

### 5.4 Internet Service Analysis

  Internet service        Churn rate
  --------------------- ------------
  DSL                         18.96%
  Fiber optic                 41.89%
  No internet service          7.40%

**Finding:** Fiber Optic customers had a considerably higher churn rate
than DSL customers.

### 5.5 Additional Services Analysis

  Service               Churn without service   Churn with service
  ------------------- ----------------------- --------------------
  Online Security                      41.77%               14.61%
  Online Backup                        39.93%               21.53%
  Device Protection                    39.13%               22.50%
  Tech Support                         41.64%               15.17%
  Streaming TV                         33.52%               30.07%
  Streaming Movies                     33.68%               29.94%

*These comparisons exclude the separate "No internet service" category.*

**Finding:** Online Security and Tech Support subscriptions were
associated with substantially lower churn rates. These are associations,
not proof that the services cause better retention.

## 6. Correlation Analysis

Calculated Pearson correlations between numerical features and the
binary churn target.

  Feature            Correlation with churn
  ---------------- ------------------------
  Tenure                             -0.352
  TotalCharges                       -0.198
  MonthlyCharges                     +0.193
  SeniorCitizen                      +0.151

Additional observation:

-   Tenure and TotalCharges had a strong positive correlation of
    `0.826`.

**Finding:** Tenure had the strongest linear correlation with churn
among the numerical features examined. Correlation does not establish
causation.

## 7. Key Business Insights

1.  Month-to-month customers represent an important retention segment.
2.  Churned customers had higher average monthly charges than retained
    customers.
3.  Electronic Check users showed the highest churn rate among payment
    methods.
4.  Fiber Optic customers showed higher churn than DSL customers.
5.  Online Security and Tech Support subscriptions were associated with
    lower churn.
6.  Tenure had the strongest numerical correlation with churn in the
    correlation analysis.

These findings will guide feature engineering and model development.
They should be validated further before being treated as causal
explanations.

## 8. Project Progress

  Workstream                             Status
  -------------------------------------- -----------------------------------------
  Google Colab and Drive setup           Completed
  Dataset collection                     Completed
  Data cleaning and validation           Completed
  Customer overview and churn analysis   Completed
  Contract and tenure analysis           Completed
  Monthly charges analysis               Completed
  Payment method analysis                Completed
  Internet service analysis              Completed
  Additional services analysis           Completed
  Correlation analysis                   Completed
  Feature engineering                    Introduced; execution not yet confirmed
  Model development                      Not started

## 9. Next Session --- Planned Tasks

1.  Continue feature engineering.
2.  Review engineered features for validity and potential data leakage.
3.  Separate predictors and target variable.
4.  Create a train-test split.
5.  Build a preprocessing pipeline for numerical and categorical
    features.
6.  Establish a baseline churn classification model.

------------------------------------------------------------------------

**Day 1 outcome:** The project environment and initial customer dataset
are prepared, data quality issues have been addressed, and the first
business-focused EDA findings have been documented.
