
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from lifelines import KaplanMeierFitter

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CLV & Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0B1120;
        color: #E5E7EB;
    }

    [data-testid="stMetric"] {
        background-color: #151F32;
        border: 1px solid #26344D;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #38BDF8;
    }

    h1, h2, h3 {
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
    }

    div[data-testid="stTabs"] button {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    telco = pd.read_csv(
        os.path.join(DATA_DIR, "telco_churn_feature_engineered.csv")
    )

    clv = pd.read_csv(
        os.path.join(DATA_DIR, "clv_predictions.csv")
    )

    segmentation = pd.read_csv(
        os.path.join(DATA_DIR, "customer_segmentation.csv")
    )

    shap_data = pd.read_csv(
        os.path.join(DATA_DIR, "shap_feature_importance.csv")
    )

    survival = pd.read_csv(
        os.path.join(DATA_DIR, "cox_hazard_ratios.csv")
    )

    return telco, clv, segmentation, shap_data, survival


@st.cache_resource
def load_models():

    churn_model = joblib.load(
        os.path.join(MODEL_DIR, "final_logistic_regression.pkl")
    )

    kmeans_model = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_segmentation_model.pkl")
    )

    return churn_model, kmeans_model


telco, clv, segmentation, shap_data, survival = load_data()
churn_model, kmeans_model = load_models()

# =====================================================
# CHURN PREDICTIONS
# =====================================================

@st.cache_data
def calculate_churn_scores(data):

    X = data.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    probabilities = churn_model.predict_proba(X)[:, 1]

    results = data[["customerID", "tenure", "MonthlyCharges",
                    "Contract", "InternetService", "Churn"]].copy()

    results["ChurnProbability"] = probabilities
    results["PredictedChurn"] = (probabilities >= 0.30).astype(int)

    return results


churn_results = calculate_churn_scores(telco)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 CLV INTELLIGENCE")

st.sidebar.caption("Customer Lifetime Value & Churn Analytics")

st.sidebar.markdown("---")

st.sidebar.markdown("### Navigation")
st.sidebar.markdown("""
- Executive Overview
- Churn Intelligence
- SHAP Explainability
- Survival Analytics
- CLV Analytics
- Customer Segmentation
- Retention ROI Simulator
""")

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio Analytics Project")
st.sidebar.caption("Models: Logistic Regression | BG/NBD | Gamma-Gamma | K-Means | Cox PH")

# =====================================================
# HEADER
# =====================================================

st.title("Customer Lifetime Value & Churn Intelligence")
st.markdown(
    "An integrated analytics platform for customer retention, "
    "churn risk assessment and revenue intelligence."
)

st.markdown("---")

# =====================================================
# TABS
# =====================================================

tabs = st.tabs([
    "Executive Overview",
    "Churn Intelligence",
    "SHAP Explainability",
    "Survival Analytics",
    "CLV Analytics",
    "Customer Segmentation",
    "ROI Simulator"
])

# =====================================================
# 1. EXECUTIVE OVERVIEW
# =====================================================

with tabs[0]:

    st.header("Executive Overview")

    total_customers = len(telco)
    actual_churn = (telco["Churn"] == "Yes").sum()
    churn_rate = actual_churn / total_customers * 100

    predicted_revenue = clv["predicted_clv_90d"].sum()
    average_clv = clv["predicted_clv_90d"].mean()

    repeat_customers = len(clv)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Telco Customers", f"{total_customers:,}")
    col2.metric("Historical Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Repeat Retail Customers", f"{repeat_customers:,}")
    col4.metric("Predicted 90-Day Revenue", f"£{predicted_revenue:,.0f}")

    st.markdown("---")

    st.subheader("Churn Model Performance")

    st.caption("Logistic Regression | Test set | Classification threshold: 0.30")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Precision", "42.72%")
    m2.metric("Recall", "92.51%")
    m3.metric("F1 Score", "58.45%")
    m4.metric("ROC-AUC", "84.19%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Historical Churn Distribution")

        churn_dist = telco["Churn"].value_counts().reset_index()
        churn_dist.columns = ["Churn", "Customers"]

        fig = px.pie(
            churn_dist,
            names="Churn",
            values="Customers",
            hole=0.55,
            title="Customer Churn Composition"
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Customer Contract Distribution")

        contract_data = (
            telco.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Customers by Contract Type"
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Churn Risk Distribution")

    risk_data = churn_results.copy()

    risk_data["Risk Category"] = pd.cut(
        risk_data["ChurnProbability"],
        bins=[0, 0.30, 0.60, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True
    )

    risk_summary = (
        risk_data["Risk Category"]
        .value_counts()
        .reindex(["Low Risk", "Medium Risk", "High Risk"])
        .reset_index()
    )

    risk_summary.columns = ["Risk Category", "Customers"]

    fig = px.bar(
        risk_summary,
        x="Risk Category",
        y="Customers",
        color="Risk Category",
        title="Model-Based Customer Risk Distribution"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Churn scores are model predictions using a 0.30 classification threshold. "
        "CLV is estimated future gross revenue for repeat retail customers, not profit."
    )

# =====================================================
# 2. CHURN INTELLIGENCE
# =====================================================

with tabs[1]:

    st.header("Churn Intelligence")

    col1, col2, col3 = st.columns(3)

    predicted_churners = churn_results["PredictedChurn"].sum()

    col1.metric("Customers Screened", f"{len(churn_results):,}")
    col2.metric("Predicted High Risk", f"{predicted_churners:,}")
    col3.metric(
        "Predicted High-Risk Share",
        f"{predicted_churners / len(churn_results) * 100:.1f}%"
    )

    st.markdown("---")

    st.subheader("Churn Probability Distribution")

    fig = px.histogram(
        churn_results,
        x="ChurnProbability",
        nbins=40,
        color="Churn",
        barmode="overlay",
        title="Predicted Churn Probability"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Risk by Contract")

    contract_risk = (
        churn_results.groupby("Contract", observed=False)
        ["ChurnProbability"].mean()
        .reset_index()
    )

    contract_risk["Average Risk (%)"] = (
        contract_risk["ChurnProbability"] * 100
    )

    fig = px.bar(
        contract_risk,
        x="Contract",
        y="Average Risk (%)",
        title="Average Predicted Churn Probability by Contract"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Risk Explorer")

    risk_filter = st.selectbox(
        "Select customer risk category",
        ["All", "High Risk", "Low Risk"]
    )

    filtered = churn_results.copy()

    if risk_filter == "High Risk":
        filtered = filtered[filtered["PredictedChurn"] == 1]

    elif risk_filter == "Low Risk":
        filtered = filtered[filtered["PredictedChurn"] == 0]

    st.dataframe(
        filtered.sort_values(
            "ChurnProbability",
            ascending=False
        ).head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Customer Risk Report",
        data=filtered.to_csv(index=False),
        file_name="customer_churn_risk_report.csv",
        mime="text/csv"
    )

    st.caption(
        "The model is designed for high-recall risk screening. "
        "False positives are possible; predictions should not trigger "
        "automatic customer actions."
    )

# =====================================================
# 3. SHAP EXPLAINABILITY
# =====================================================

with tabs[2]:

    st.header("Model Explainability")

    st.write(
        "Global SHAP importance measures the average magnitude of each "
        "feature's contribution to the model output."
    )

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=30,
        value=15
    )

    top_shap = shap_data.head(top_n).sort_values(
        "Mean_Absolute_SHAP"
    )

    fig = px.bar(
        top_shap,
        x="Mean_Absolute_SHAP",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Global SHAP Features"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance Table")

    st.dataframe(
        shap_data.head(top_n),
        use_container_width=True
    )

    image_path = os.path.join(
        os.path.dirname(__file__),
        "shap_feature_importance.png"
    )

    if os.path.exists(image_path):
        st.image(
            image_path,
            caption="Saved Global SHAP Feature Importance"
        )

    st.info(
        "SHAP explains model behaviour, not causal relationships. "
        "Mean absolute SHAP values show importance magnitude, not direction."
    )

# =====================================================
# 4. SURVIVAL ANALYTICS
# =====================================================

with tabs[3]:

    st.header("Customer Survival Analytics")

    st.write(
        "Kaplan-Meier analysis estimates the proportion of customers "
        "remaining before the recorded churn event over tenure."
    )

    km_data = telco[["tenure", "Churn"]].copy()

    km_data["Event"] = (km_data["Churn"] == "Yes").astype(int)

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=km_data["tenure"],
        event_observed=km_data["Event"],
        label="All Customers"
    )

    survival_curve = kmf.survival_function_.reset_index()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=survival_curve["timeline"],
            y=survival_curve["All Customers"],
            mode="lines",
            name="Customer Survival"
        )
    )

    fig.update_layout(
        title="Kaplan-Meier Customer Survival Curve",
        xaxis_title="Customer Tenure (Months)",
        yaxis_title="Estimated Survival Probability",
        yaxis=dict(tickformat=".0%"),
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Survival by Contract Type")

    fig = go.Figure()

    for contract in telco["Contract"].unique():

        subset = telco[telco["Contract"] == contract]

        km = KaplanMeierFitter()

        km.fit(
            subset["tenure"],
            event_observed=(subset["Churn"] == "Yes"),
            label=contract
        )

        curve = km.survival_function_

        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.iloc[:, 0],
                mode="lines",
                name=contract
            )
        )

    fig.update_layout(
        title="Survival Probability by Contract",
        xaxis_title="Tenure (Months)",
        yaxis_title="Survival Probability",
        yaxis=dict(tickformat=".0%"),
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cox Proportional Hazards Results")

    selected = survival[
        ["covariate", "exp(coef)", "p",
         "exp(coef) lower 95%", "exp(coef) upper 95%"]
    ].copy()

    st.dataframe(
        selected.sort_values("exp(coef)", ascending=False),
        use_container_width=True
    )

    st.info(
        "Cox model concordance index during training: 0.8257. "
        "Some proportional-hazards diagnostics were previously flagged. "
        "Hazard ratios should therefore be interpreted cautiously."
    )

    st.caption(
        "This is exploratory survival analysis based on the available "
        "cross-sectional customer tenure and churn status, not a fully "
        "observed customer lifecycle history."
    )

# =====================================================
# 5. CLV ANALYTICS
# =====================================================

with tabs[4]:

    st.header("Customer Lifetime Value Analytics")

    total_clv = clv["predicted_clv_90d"].sum()
    mean_clv = clv["predicted_clv_90d"].mean()
    median_clv = clv["predicted_clv_90d"].median()

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted 90-Day Revenue", f"£{total_clv:,.0f}")
    col2.metric("Average Predicted CLV", f"£{mean_clv:,.2f}")
    col3.metric("Median Predicted CLV", f"£{median_clv:,.2f}")

    st.markdown("---")

    st.subheader("CLV Distribution")

    fig = px.histogram(
        clv,
        x="predicted_clv_90d",
        nbins=50,
        title="Predicted 90-Day Customer Value"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 Customers by Predicted CLV")

    top_clv = (
        clv.nlargest(20, "predicted_clv_90d")
        .sort_values("predicted_clv_90d", ascending=True)
        .copy()
    )

    top_clv["Customer ID"] = top_clv["Customer ID"].astype(str)

    fig = px.bar(
        top_clv,
        x="predicted_clv_90d",
        y="Customer ID",
        orientation="h",
        text="predicted_clv_90d",
        title="Top 20 Customers by Predicted 90-Day CLV",
        labels={
            "predicted_clv_90d": "Predicted CLV (£)",
            "Customer ID": "Customer ID"
        }
    )

    fig.update_traces(
        texttemplate="£%{x:,.0f}",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=100, r=100, t=70, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CLV Customer Explorer")

    st.dataframe(
        clv.sort_values(
            "predicted_clv_90d",
            ascending=False
        ).head(100),
        use_container_width=True
    )

    st.download_button(
        "Download CLV Predictions",
        data=clv.to_csv(index=False),
        file_name="customer_clv_predictions.csv",
        mime="text/csv"
    )

    st.caption(
        "CLV represents estimated 90-day gross revenue for repeat purchasers. "
        "It is not profit, and individual-level forecast accuracy was limited "
        "in historical validation."
    )

# =====================================================
# 6. CUSTOMER SEGMENTATION
# =====================================================

with tabs[5]:

    st.header("Customer Segmentation")

    segment_summary = (
        segmentation.groupby("Segment")
        .agg(
            Customers=("Customer ID", "count"),
            Revenue=("Monetary", "sum"),
            AverageRevenue=("Monetary", "mean"),
            AverageFrequency=("Frequency", "mean"),
            AverageRecency=("Recency", "mean")
        )
        .reset_index()
    )

    segment_summary["Revenue Share (%)"] = (
        segment_summary["Revenue"] /
        segment_summary["Revenue"].sum() * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            segment_summary,
            names="Segment",
            values="Customers",
            hole=0.5,
            title="Customer Distribution by Segment"
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            segment_summary,
            x="Segment",
            y="Revenue",
            color="Segment",
            title="Historical Revenue by Segment"
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Performance")

    st.dataframe(
        segment_summary.sort_values(
            "Revenue",
            ascending=False
        ),
        use_container_width=True
    )

    st.subheader("Customer Behaviour")

    fig = px.scatter(
        segmentation,
        x="Frequency",
        y="Monetary",
        color="Segment",
        size="AverageOrderValue",
        hover_data=["Customer ID", "Recency"],
        log_x=True,
        log_y=True,
        title="Customer Frequency vs Monetary Value"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    selected_segment = st.selectbox(
        "Explore a customer segment",
        sorted(segmentation["Segment"].unique())
    )

    segment_customers = segmentation[
        segmentation["Segment"] == selected_segment
    ]

    st.dataframe(
        segment_customers.head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Segment Customers",
        data=segment_customers.to_csv(index=False),
        file_name=f"{selected_segment.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

    st.caption(
        "K-Means clusters are exploratory behavioural segments, "
        "not verified churn categories."
    )

# =====================================================
# 7. RETENTION ROI SIMULATOR
# =====================================================

with tabs[6]:

    st.header("Retention ROI Simulator")

    st.write(
        "Estimate the potential financial impact of a retention campaign "
        "using customer CLV and user-defined campaign assumptions."
    )

    col1, col2 = st.columns(2)

    with col1:

        target_customers = st.number_input(
            "Customers targeted",
            min_value=1,
            max_value=100000,
            value=500
        )

        retention_rate = st.slider(
            "Expected retention improvement (%)",
            min_value=1,
            max_value=100,
            value=10
        )

    with col2:

        campaign_cost = st.number_input(
            "Campaign cost per targeted customer (£)",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        value_per_customer = st.number_input(
            "Expected revenue value per retained customer (£)",
            min_value=0.0,
            value=float(median_clv),
            step=10.0
        )

    retained_customers = target_customers * retention_rate / 100

    revenue_saved = retained_customers * value_per_customer

    total_campaign_cost = target_customers * campaign_cost

    net_return = revenue_saved - total_campaign_cost

    roi = (
        (net_return / total_campaign_cost) * 100
        if total_campaign_cost > 0
        else 0
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Expected Retained Customers", f"{retained_customers:,.0f}")
    col2.metric("Estimated Gross Revenue Retained", f"£{revenue_saved:,.2f}")
    col3.metric("Campaign Spend", f"£{total_campaign_cost:,.2f}")
    col4.metric("Revenue-Based ROI", f"{roi:,.1f}%")

    st.subheader("Campaign Financial Summary")

    fig = go.Figure(
        data=[
            go.Bar(
                x=[
                    "Gross Revenue Retained",
                    "Campaign Spend",
                    "Revenue Less Campaign Spend"
                ],
                y=[revenue_saved, total_campaign_cost, net_return]
            )
        ]
    )

    fig.update_layout(
        title="Estimated Retention Campaign Impact",
        yaxis_title="Amount (£)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    if total_campaign_cost == 0:
        st.info("Enter a campaign cost greater than zero to calculate a meaningful ROI.")
    elif roi > 0:
        st.success("The scenario generates more estimated gross revenue than campaign spend.")
    else:
        st.warning("The estimated gross revenue retained is lower than campaign spend.")

    st.caption(
        "Illustrative scenario only. Retention improvement, campaign costs "
        "and revenue value are user assumptions. Revenue less campaign spend "
        "is not accounting profit: product costs, operating expenses and "
        "gross margin are not included. ROI is revenue-based, not profit-based. "
        "The simulator does not establish causal campaign effectiveness."
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Customer Lifetime Value & Churn Intelligence System | "
    "Exploratory Portfolio Project | "
    "Results are estimates and should not be treated as guaranteed outcomes."
)
