# Customer Lifetime Value & Churn Intelligence System

An interactive machine-learning dashboard combining customer churn risk
analysis, model explainability, survival analytics, customer lifetime
value (CLV) forecasting, behavioural segmentation, and retention
campaign scenario modelling.

**Live application:** [Open the CLV & Churn Intelligence
Dashboard](https://clv-churn-intelligence.streamlit.app/)

------------------------------------------------------------------------

## Overview

This project brings several customer analytics workflows into one
Streamlit application. It explores questions such as:

-   Which customers are most at risk of churn?
-   Which features influence the churn model's predictions?
-   How do customer tenure and retention patterns relate?
-   Which customers have the highest predicted 90-day value?
-   How do behavioural customer segments differ?
-   Under stated assumptions, how might a retention campaign's estimated
    gross revenue compare with its campaign spend?

The application is an exploratory portfolio project. Its outputs are
estimates and decision-support signals, not guaranteed outcomes or
automated business decisions.

## Dashboard Modules

  -----------------------------------------------------------------------
  Module                              Description
  ----------------------------------- -----------------------------------
  Executive Overview                  Headline KPIs and churn model
                                      performance

  Churn Intelligence                  Customer-level churn risk scoring
                                      and exploration

  SHAP Explainability                 Feature-level explanations of churn
                                      predictions

  Survival Analytics                  Customer retention and survival
                                      analysis

  CLV Analytics                       Predicted 90-day customer value and
                                      customer exploration

  Customer Segmentation               Behavioural customer groups and
                                      segment analysis

  ROI Simulator                       Illustrative retention campaign
                                      scenario analysis
  -----------------------------------------------------------------------

## Methodology & Results

### Churn Prediction

A Logistic Regression model estimates customer churn risk. The deployed
model is loaded from `models/final_logistic_regression.pkl`.

Test-set performance at a classification threshold of **0.30**:

  Metric         Score
  ----------- --------
  Precision     42.72%
  Recall        92.51%
  F1-score      58.45%
  ROC-AUC       84.19%

The threshold favours identifying a high proportion of churn cases. The
trade-off is a higher number of false positives. The model is best
interpreted as a broad retention-risk screening tool, not as a basis for
automatic intervention.

### Model Explainability

SHAP feature importance is used to examine which input features
contribute most to model predictions. These explanations describe model
behaviour; they do not establish that a feature causes churn.

### Survival Analytics

Survival analysis explores customer retention over time. The Cox model's
reported concordance index is **0.8257**.

Some proportional-hazards diagnostics were flagged during analysis. Cox
estimates should therefore be interpreted as exploratory rather than
definitive evidence that the proportional-hazards assumption holds.

### Customer Lifetime Value

The CLV workflow uses the BG/NBD and Gamma-Gamma modelling approach on
the Online Retail II transaction dataset to estimate purchasing activity
and average order value, then derive predicted 90-day CLV.

Current analysis summary:

-   Customers in the retail analysis: **5,878**
-   Customers with repeat-purchase CLV predictions: **4,189**
-   Mean predicted 90-day CLV: **£552.18**
-   Median predicted 90-day CLV: **£209.86**
-   Total predicted 90-day gross revenue: **£2,313,096.85**

These are model-based estimates, not realised revenue. The total
represents gross revenue, not profit. Individual-level forecast
validation is limited, so predictions should be interpreted with that
limitation in mind.

### Customer Segmentation

K-means clustering groups customers based on behavioural features. Four
clusters were selected for interpretability. Segment labels are
descriptive behavioural interpretations, not verified churn outcomes or
fixed customer attributes.

### Retention ROI Simulation

The ROI Simulator accepts user-defined assumptions for target audience
size, expected retention improvement, campaign cost per targeted
customer, and expected revenue value per retained customer.

The displayed ROI is **revenue-based, not profit-based**. The
calculation excludes product costs, operating expenses, gross margin,
and causal measurement of campaign effectiveness.

## Technology Stack

-   Python
-   Pandas and NumPy --- data processing
-   Scikit-learn --- machine learning and clustering
-   SHAP --- model explainability during analysis
-   Lifelines --- survival analysis
-   Plotly --- interactive visualisations
-   Streamlit --- dashboard application
-   Joblib --- persisted model loading

## Repository Structure

``` text
Customer-Lifetime-Value-Churn-Intelligence/
├── dashboard/
│   ├── app.py
│   └── shap_feature_importance.png
├── data/
│   ├── telco_churn_raw.csv
│   ├── telco_churn_processed.csv
│   ├── telco_churn_feature_engineered.csv
│   ├── customer_segmentation.csv
│   ├── clv_predictions.csv
│   ├── shap_feature_importance.csv
│   └── cox_hazard_ratios.csv
├── models/
│   ├── final_logistic_regression.pkl
│   ├── kmeans_segmentation_model.pkl
│   ├── customer_segmentation_scaler.pkl
│   ├── cox_survival_model.pkl
│   ├── bgnbd_parameters.json
│   ├── gamma_gamma_parameters.json
│   └── cox_model_parameters.json
├── notebooks/
├── daily progress/
├── requirements.txt
└── README.md
```

The repository may contain additional analysis files and model
artefacts. The dashboard's runtime dependencies are the files it loads
directly.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/20deepak28/Customer-Lifetime-Value-Churn-Intelligence.git
cd Customer-Lifetime-Value-Churn-Intelligence
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the dashboard

``` bash
streamlit run dashboard/app.py
```

## Deployment

The application is deployed on Streamlit Community Cloud:

**<https://clv-churn-intelligence.streamlit.app/>**

The Streamlit entry point is `dashboard/app.py`. Dependencies are
specified in the root-level `requirements.txt`.

## Limitations & Responsible Use

-   Performance metrics are based on a particular test split and
    threshold; results may not generalise to other customer populations
    or time periods.
-   High recall at the selected churn threshold comes with lower
    precision and more false positives.
-   SHAP values explain model predictions, not causal relationships.
-   Survival-model diagnostics raised proportional-hazards concerns; Cox
    results are exploratory.
-   CLV forecasts estimate gross revenue, not profit or guaranteed
    future spending.
-   Customer segment labels are descriptive interpretations of clusters.
-   ROI outputs depend on user-entered assumptions and are not causal
    estimates of campaign effectiveness.
-   Model outputs should not be the sole basis for decisions that
    materially affect customers.

## Author

**Deepak Kumar Shukla**

-   GitHub: [20deepak28](https://github.com/20deepak28)
-   Live project: [CLV & Churn Intelligence
    Dashboard](https://clv-churn-intelligence.streamlit.app/)

------------------------------------------------------------------------

*Built as a machine-learning and business analytics portfolio project.*
