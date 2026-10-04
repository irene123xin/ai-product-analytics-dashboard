"""Streamlit dashboard for the synthetic AI SaaS product analytics case study."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import churn_comparison, cohort_retention, conversion_funnel, executive_metrics, load_users, product_recommendations
from src.model import train_churn_models

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "product_users.csv"
ACCENT = "#087E6B"
COLORS = [ACCENT, "#315B73", "#7A8B8B", "#B7C2BE", "#D6A85F", "#9B5C5C"]

st.set_page_config(page_title="AI Product Analytics", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    :root { --accent: #087E6B; --ink: #172321; --muted: #5F6F6B; --line: #DCE4E1; --surface: #F5F8F7; }
    .stApp { background: #FFFFFF; color: var(--ink); }
    .block-container { max-width: 1380px; padding-top: 2.1rem; padding-bottom: 4rem; }
    h1, h2, h3 { letter-spacing: -0.025em; color: var(--ink); }
    h1 { font-size: 2.35rem !important; font-weight: 660 !important; }
    h2 { margin-top: 1.8rem !important; font-size: 1.55rem !important; }
    [data-testid="stMetric"] { border-top: 3px solid var(--accent); padding: 1rem 0.35rem 0.7rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    .section-note { color: var(--muted); max-width: 72ch; margin-bottom: 1.25rem; }
    .recommendation { border-top: 1px solid var(--line); padding: 1rem 0 1.15rem; }
    .recommendation h4 { margin: 0 0 .35rem; color: var(--ink); }
    .recommendation p { margin: .25rem 0; color: var(--muted); }
    div[data-baseweb="select"] > div { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_users(str(DATA_PATH))


@st.cache_resource
def get_models(df: pd.DataFrame):
    return train_churn_models(df)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    plan = st.sidebar.multiselect("Plan type", sorted(df["plan_type"].unique()), default=sorted(df["plan_type"].unique()))
    channel = st.sidebar.multiselect("Acquisition channel", sorted(df["acquisition_channel"].unique()), default=sorted(df["acquisition_channel"].unique()))
    churn_label = st.sidebar.selectbox("Churn status", ["All", "Retained", "Churned"])
    mask = df["plan_type"].isin(plan) & df["acquisition_channel"].isin(channel)
    if churn_label != "All":
        mask &= df["churned"].eq(1 if churn_label == "Churned" else 0)
    st.sidebar.caption("Synthetic portfolio dataset. Filters apply to analysis pages; the model is trained on the full dataset.")
    return df.loc[mask].copy()


def plot(fig):
    fig.update_layout(template="plotly_white", colorway=COLORS, margin=dict(l=15, r=15, t=45, b=15), font=dict(family="Arial", color="#263532"), legend_title_text="")
    st.plotly_chart(fig, width="stretch")


df = get_data()
filtered = filter_data(df)
pages = ["Overview", "User Engagement", "Conversion Funnel", "Retention", "Churn Analysis", "Churn Prediction", "Product Recommendations"]
requested_page = st.query_params.get("page", "Overview")
default_page = pages.index(requested_page) if requested_page in pages else 0
page = st.sidebar.radio("Analysis", pages, index=default_page)
if page != requested_page:
    st.query_params["page"] = page

st.caption("PRODUCT ANALYTICS CASE STUDY")
st.title("AI Product Analytics Dashboard")
st.markdown('<p class="section-note">Engagement, activation, retention, and churn signals for a synthetic AI collaboration product.</p>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No users match the selected filters. Broaden the filter selection to continue.")
    st.stop()

if page == "Overview":
    metrics = executive_metrics(filtered)
    cols = st.columns(5)
    cols[0].metric("Users", f"{metrics['total_users']:,}")
    cols[1].metric("Active in 30 days", f"{metrics['active_users']:,}")
    cols[2].metric("Paid conversion", f"{metrics['paid_conversion_rate']:.1%}")
    cols[3].metric("Churn rate", f"{metrics['churn_rate']:.1%}")
    cols[4].metric("ARPU", f"${metrics['arpu']:.2f}")
    left, right = st.columns([1.2, 1])
    with left:
        monthly = filtered.assign(signup_month=filtered["signup_date"].dt.to_period("M").astype(str)).groupby("signup_month", as_index=False).agg(users=("user_id", "count"), paid=("trial_to_paid", "sum"))
        plot(px.line(monthly, x="signup_month", y=["users", "paid"], markers=True, title="Monthly acquisition and conversion"))
    with right:
        plan = filtered.groupby("plan_type", as_index=False).agg(users=("user_id", "count"), revenue=("monthly_revenue", "sum"))
        plot(px.bar(plan, x="plan_type", y="revenue", color="plan_type", title="Monthly revenue by plan"))
    st.info(f"The most-used product capability in this selection is **{metrics['most_used_feature']}**.")

elif page == "User Engagement":
    st.header("User engagement")
    st.markdown('<p class="section-note">Usage frequency, depth, and feature adoption across commercial and acquisition segments.</p>', unsafe_allow_html=True)
    group = filtered.groupby("plan_type", as_index=False).agg(sessions=("sessions_30d", "mean"), active_days=("active_days_30d", "mean"), session_minutes=("avg_session_minutes", "mean"))
    plot(px.bar(group, x="plan_type", y=["sessions", "active_days"], barmode="group", title="Average engagement by plan"))
    c1, c2 = st.columns(2)
    with c1:
        feature = pd.DataFrame({"Feature": ["AI generation", "Export", "Collaboration"], "Adoption": [(filtered[c] > 0).mean() for c in ["ai_feature_usage", "export_feature_usage", "collaboration_feature_usage"]]})
        fig = px.bar(feature, x="Adoption", y="Feature", orientation="h", title="Feature adoption")
        fig.update_xaxes(tickformat=".0%")
        plot(fig)
    with c2:
        channel = filtered.groupby("acquisition_channel", as_index=False).agg(sessions=("sessions_30d", "mean"), active_days=("active_days_30d", "mean")).sort_values("sessions")
        plot(px.scatter(channel, x="active_days", y="sessions", text="acquisition_channel", size="sessions", title="Engagement by acquisition channel"))

elif page == "Conversion Funnel":
    st.header("Conversion funnel")
    funnel = conversion_funnel(filtered)
    fig = go.Figure(go.Funnel(y=funnel["stage"], x=funnel["users"], textinfo="value+percent initial"))
    fig.update_traces(marker_color=COLORS[: len(funnel)])
    plot(fig)
    st.dataframe(funnel.assign(overall_conversion=funnel["overall_conversion"].map("{:.1%}".format), step_conversion=funnel["step_conversion"].map("{:.1%}".format)), hide_index=True, width="stretch")

elif page == "Retention":
    st.header("Cohort retention")
    st.markdown('<p class="section-note">A user is retained in a month when their observed last-active month is at or beyond that interval. Recent cohorts appear only where sufficient time has elapsed.</p>', unsafe_allow_html=True)
    matrix = cohort_retention(filtered).tail(12)
    fig = px.imshow(matrix, text_auto=".0%", aspect="auto", color_continuous_scale=[[0, "#EDF3F1"], [1, ACCENT]], zmin=0, zmax=1, title="Signup cohort retention")
    plot(fig)

elif page == "Churn Analysis":
    st.header("Churn analysis")
    comparison = churn_comparison(filtered).reset_index(names="Metric")
    st.dataframe(comparison.style.format({"Retained": "{:.2f}", "Churned": "{:.2f}", "Difference": "{:+.2f}"}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        plan = filtered.groupby("plan_type", as_index=False)["churned"].mean()
        fig = px.bar(plan, x="plan_type", y="churned", color="plan_type", title="Churn rate by plan")
        fig.update_yaxes(tickformat=".0%")
        plot(fig)
    with c2:
        channel = filtered.groupby("acquisition_channel", as_index=False)["churned"].mean().sort_values("churned")
        fig = px.bar(channel, x="churned", y="acquisition_channel", orientation="h", title="Churn rate by acquisition channel")
        fig.update_xaxes(tickformat=".0%")
        plot(fig)

elif page == "Churn Prediction":
    st.header("Churn prediction")
    st.markdown('<p class="section-note">Two interpretable baselines estimate churn association from observed product behavior. Metrics are held-out test results, not evidence of causal effects.</p>', unsafe_allow_html=True)
    results = get_models(df)
    st.dataframe(results.metrics.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        matrix = results.confusion_matrices[results.best_model_name]
        plot(px.imshow(matrix, text_auto=True, x=["Predicted retained", "Predicted churn"], y=["Actual retained", "Actual churn"], color_continuous_scale=[[0, "#EDF3F1"], [1, ACCENT]], title=f"Confusion matrix: {results.best_model_name}"))
    with c2:
        importance = results.feature_importance.sort_values("importance")
        plot(px.bar(importance, x="importance", y="feature", orientation="h", title="Leading model signals"))
    st.caption("Feature importance shows predictive association within this synthetic dataset. It does not establish that changing a feature will cause churn to change.")

else:
    st.header("Product recommendations")
    st.markdown('<p class="section-note">Prioritized opportunities derived from descriptive patterns. Each should be validated through instrumentation and controlled experiments.</p>', unsafe_allow_html=True)
    for rec in product_recommendations(filtered):
        st.markdown(f'<div class="recommendation"><h4>{rec["title"]}</h4><p><strong>Finding:</strong> {rec["finding"]}</p><p><strong>Proposed action:</strong> {rec["action"]}</p><p><strong>Expected impact:</strong> {rec["impact"]}</p></div>', unsafe_allow_html=True)
