# Customer Lifetime Value and Churn Intelligence System

## Daily Progress Log --- Machine Learning Model Development

**Date:** 15 November 2026\
**Dataset:** IBM Telco Customer Churn\
**Notebook:** `04_Machine_Learning_Model_Development.ipynb`

------------------------------------------------------------------------

## 1. Train, Validation, and Test Split

The 7,043-row dataset was divided into training and test sets. The
training portion was then split again to create a validation set.

  ------------------------------------------------------------------------
  Dataset                                    Records Purpose
  --------------------- ---------------------------- ---------------------
  Model training                               4,507 Fit model parameters
  (`X_fit`)                                          

  Validation (`X_val`)                         1,127 Compare models and
                                                     select a decision
                                                     threshold

  Test (`X_test`)                              1,409 Holdout evaluation
  ------------------------------------------------------------------------

Stratification was used to preserve churn/non-churn proportions.

## 2. Models Trained

Three classification pipelines were developed:

1.  Logistic Regression
2.  Random Forest Classifier
3.  LightGBM Classifier

The preprocessing pipeline used `StandardScaler` for numerical columns
and `OneHotEncoder(handle_unknown='ignore')` for categorical columns,
combined using `ColumnTransformer`.

## 3. Validation-Based Threshold Comparison

The models were retrained using the 4,507-row model-training subset.
Predicted probabilities were evaluated on the 1,127-row validation
subset.

  Model                   Threshold   Precision      Recall    F1 Score
  --------------------- ----------- ----------- ----------- -----------
  Logistic Regression          0.30       0.417   **0.913**       0.573
  Random Forest                0.45   **0.586**       0.706   **0.640**
  LightGBM                     0.40       0.530       0.759       0.624

### Interpretation and selection

-   Logistic Regression at threshold 0.30 achieved the highest recall
    among the compared validation configurations.
-   Random Forest at threshold 0.45 achieved the highest precision and
    F1 score in the supplied validation results.
-   LightGBM at threshold 0.40 provided an intermediate precision-recall
    balance.

The project's stated business priority is to reduce missed churners
(false negatives), because missing a customer who is about to leave is
considered more costly than contacting a customer who would have stayed.

Based on that priority, Logistic Regression with a provisional threshold
of **0.30** was selected for holdout evaluation.

## 4. Holdout Evaluation --- Logistic Regression at Threshold 0.30

The selected Logistic Regression pipeline was evaluated on the 1,409-row
test set.

  Metric            Result
  ----------- ------------
  Accuracy          64.80%
  Precision         42.51%
  Recall        **92.51%**
  F1 Score          58.25%
  ROC-AUC           84.32%

### Confusion matrix

  Actual vs Predicted     Predicted Stayed   Predicted Churned
  --------------------- ------------------ -------------------
  Actual Stayed                        567                 468
  Actual Churned                        28                 346

### Interpretation

-   346 of the 374 actual churners were correctly identified.
-   28 actual churners were missed.
-   468 customers who stayed were incorrectly flagged as potential
    churners.
-   567 customers who stayed were correctly classified.

The model achieved high churn recall, consistent with the project's
priority. However, precision of 42.51% means many flagged customers did
not actually churn. A retention team would need to consider the cost and
capacity implications of contacting this many customers.

## 5. Validation and Holdout Metric Comparison

  Metric        Validation   Holdout test
  ----------- ------------ --------------
  Precision          41.7%         42.51%
  Recall             91.3%         92.51%
  F1 Score           57.3%         58.25%

The scores are similar, which is encouraging. However, the test set had
been inspected during earlier model and threshold experiments before the
validation split was introduced. Therefore, it was not completely
untouched throughout the full development process.

For a stronger final performance claim, a future iteration should
reserve a fresh holdout set or use nested cross-validation for model and
threshold selection, followed by one final evaluation.

## 6. Business Takeaways

1.  The selected model prioritises identifying as many potential
    churners as possible.
2.  High recall comes with a substantial number of false positives.
3.  The threshold should ultimately be chosen using business costs,
    including retention-offer costs, expected value of a saved customer,
    and retention-team capacity.
4.  Model predictions indicate risk; they do not prove that a customer
    will churn or that a retention offer will prevent churn.

## 7. Files and Model Status

The baseline Logistic Regression model had previously been saved as:

`models/logistic_regression_baseline.pkl`

A final-model save was proposed after holdout evaluation, using:

`models/final_logistic_regression.pkl`

**Status:** The final-model save command was provided, but successful
execution was not yet confirmed.

## 8. Next Steps

1.  Save and verify the selected Logistic Regression pipeline.
2.  Start `05_SHAP_Model_Explainability.ipynb`.
3.  Use SHAP to investigate global feature importance and individual
    customer-level explanations.
4.  Distinguish predictive associations from causal effects.
5.  Revisit the evaluation design and obtain a genuinely fresh holdout
    evaluation before making final performance claims.

------------------------------------------------------------------------

**Daily milestone:** Trained and compared Logistic Regression, Random
Forest, and LightGBM; used validation data to select a recall-focused
Logistic Regression configuration; and evaluated that configuration on
the holdout dataset.
