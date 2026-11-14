# Customer Lifetime Value and Churn Intelligence System

## Feature Engineering Progress Log

**Date:** 14 November 2026\
**Dataset:** IBM Telco Customer Churn\
**Notebook:** `02_EDA_Feature_Engineering.ipynb`

------------------------------------------------------------------------

## 1. Feature Engineering

### 1.1 TenureGroup

Customers were grouped by tenure using these bins:

``` python
ml_df['TenureGroup'] = pd.cut(
    ml_df['tenure'],
    bins=[-1, 12, 24, 48, 72],
    labels=['New', 'Developing', 'Established', 'Loyal']
)
```

The first bin begins at `-1` so customers with zero months of tenure are
included in `New`. With pandas' default right-inclusive intervals, the
groups are:

-   New: 0--12 months
-   Developing: 13--24 months
-   Established: 25--48 months
-   Loyal: 49--72 months

#### TenureGroup vs Churn

  Tenure Group     Retained (%)   Churned (%)
  -------------- -------------- -------------
  New                     52.56         47.44
  Developing              71.29         28.71
  Established             79.61         20.39
  Loyal                   90.49          9.51

**Finding:** Churn decreases as tenure group increases. New customers
have the highest observed churn rate (47.44%), while loyal customers
have the lowest (9.51%).

**Business interpretation:** The first 12 months appear to be a critical
retention period. Possible actions include improving onboarding,
engaging customers early, providing proactive support, and identifying
dissatisfaction before customers leave.

This is an association in the dataset, not proof that tenure causes
retention.

### 1.2 AdditionalServicesCount

This feature counts the number of additional services marked `Yes` for
each customer.

``` python
service_columns = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

ml_df['AdditionalServicesCount'] = (
    ml_df[service_columns] == 'Yes'
).sum(axis=1)
```

The feature ranges from 0 to 6.

#### Customer distribution

    Additional Services   Customers   Dataset Share (%)
  --------------------- ----------- -------------------
                      0       2,219                31.5
                      1         966                13.7
                      2       1,033                14.7
                      3       1,118                15.9
                      4         852                12.1
                      5         571                 8.1
                      6         284                 4.0
              **Total**   **7,043**           **100.0**

#### AdditionalServicesCount vs Churn

Approximate churn rates read from the generated chart:

    Additional Services   Approximate Churn Rate (%)
  --------------------- ----------------------------
                      0                           21
                      1                           45
                      2                           36
                      3                           27
                      4                           22
                      5                           12
                      6                            5

**Finding:** Customers with exactly one additional service have the
highest observed churn rate (approximately 45%). Churn generally
declines among customers with more additional services, reaching
approximately 5% for customers with all six.

**Business interpretation:** Customers with limited service adoption may
be candidates for targeted engagement and relevant cross-selling. This
pattern is associative; it does not establish that adding services
causes customers to stay.

*Note: These churn rates are approximate values read from the chart, not
exact computed percentages.*

------------------------------------------------------------------------

## 2. Data Quality and Validation

  Check                                                                   Result
  ---------------------------------------------------- -------------------------
  Original dataset rows                                                    7,043
  Feature-engineered dataset shape                       7,043 rows × 22 columns
  Missing values in `TenureGroup`                                              0
  Missing values in `AdditionalServicesCount`                                  0
  Duplicate rows in original dataset                                           0
  Duplicate `customerID` values in original dataset                            0
  Duplicate rows in `ml_df` after identifier removal                          22

### Explanation of the 22 repeated feature rows

The original dataset contains no duplicate records and no duplicate
customer IDs. The 22 repeated rows appeared in `ml_df` after
`customerID` was removed.

Different customers can have identical values across all remaining
feature columns. Therefore, these are identical feature profiles, not
confirmed duplicate customer records.

**Decision:** Retain all 7,043 observations. Do not delete these rows
solely because their non-ID feature values match.

------------------------------------------------------------------------

## 3. Feature Engineering Summary

Two engineered features have been created:

1.  `TenureGroup`
2.  `AdditionalServicesCount`

Both have zero missing values. Initial analysis suggests that both are
associated with customer churn and may provide useful information for
predictive modelling. Their contribution must be evaluated on held-out
test data.

The working feature-engineered dataset contains 7,043 rows and 22
columns.

------------------------------------------------------------------------

## 4. Next Steps

1.  Save the feature-engineered dataset as
    `telco_churn_feature_engineered.csv`.
2.  Define `Churn` as the prediction target and separate it from input
    features.
3.  Split the data into training and test sets using stratification.
4.  Build a preprocessing pipeline for numerical and categorical
    features.
5.  Train Logistic Regression as the baseline model.
6.  Evaluate using precision, recall, F1-score, ROC-AUC, and PR-AUC.
7.  Compare additional models after establishing the baseline.

**Modelling practice:** Fit preprocessing steps only on training data
through a pipeline to prevent data leakage. Keep the 22 legitimate
observations whose feature profiles happen to match.
