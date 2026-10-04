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
NEGATIVE = "#A34F4F"
COLORS = [ACCENT, "#3D6A63", "#78928C", "#A8B8B4", "#C7A86E", NEGATIVE]
px.defaults.color_discrete_sequence = COLORS

st.set_page_config(page_title="AI Product Analytics", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    :root {
        --accent: #087E6B;
        --accent-soft: #E7F1EE;
        --ink: #192522;
        --muted: #66736F;
        --line: #DDE5E1;
        --surface: #F4F6F4;
        --panel: #FFFFFF;
        --negative: #A34F4F;
    }
    html { font-size: 16px; }
    .stApp { background: #F8F9F7; color: var(--ink); }
    .block-container {
        max-width: 1420px;
        padding: 2rem 2.25rem 4.5rem;
    }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.022em; }
    h1 {
        font-size: clamp(2rem, 3vw, 2.65rem) !important;
        font-weight: 650 !important;
        line-height: 1.08 !important;
        margin-bottom: .55rem !important;
    }
    h2 {
        font-size: 1.45rem !important;
        font-weight: 620 !important;
        margin: 2rem 0 .45rem !important;
    }
    h3 { font-size: 1.08rem !important; font-weight: 600 !important; }
    [data-testid="stMetric"] {
        min-height: 7.25rem;
        padding: 1.05rem 1.1rem .9rem;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(39, 62, 56, .035);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: .82rem;
        font-weight: 540;
        letter-spacing: .01em;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.75rem;
        font-weight: 650;
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stSidebar"] {
        background: #F1F4F1;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.25rem; }
    [data-testid="stSidebar"] h2 {
        font-size: 1rem !important;
        margin: 1.35rem 0 .55rem !important;
    }
    [data-testid="stSidebar"] label { color: #35423F; font-size: .9rem; }
    [data-baseweb="tag"] {
        background-color: #E2E9E6 !important;
        color: #31413D !important;
        border-radius: 5px !important;
    }
    div[data-baseweb="select"] > div {
        background: #FAFBFA;
        border-color: #D6DFDB;
        border-radius: 7px;
        box-shadow: none;
    }
    .hero-kicker {
        margin: 0 0 .45rem;
        color: var(--accent);
        font-size: .74rem;
        font-weight: 650;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero-subtitle, .section-note {
        color: var(--muted);
        max-width: 74ch;
        line-height: 1.55;
    }
    .hero-subtitle { margin: 0 0 1.75rem; font-size: 1rem; }
    .section-note { margin: 0 0 1.25rem; }
    .insight-line {
        margin: 1rem 0 1.35rem;
        padding: .8rem 1rem;
        color: #29423C;
        background: var(--accent-soft);
        border-left: 3px solid var(--accent);
        border-radius: 0 7px 7px 0;
        font-size: .92rem;
    }
    .recommendation {
        margin: 0 0 .85rem;
        padding: 1.15rem 1.25rem 1.2rem;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    .recommendation h4 { margin: 0 0 .75rem; font-size: 1.08rem; }
    .recommendation p { margin: .4rem 0; color: var(--muted); line-height: 1.48; }
    .recommendation strong { color: #34433F; font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 8px; }
    hr { border-color: var(--line); }
    @media (max-width: 900px) {
        .block-container { padding: 1.35rem 1rem 3rem; }
        [data-testid="stMetric"] { min-height: 6.5rem; }
    }
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
    fig.update_layout(
        template="plotly_white",
        colorway=COLORS,
        margin=dict(l=18, r=12, t=58, b=22),
        height=390,
        font=dict(family="Arial, sans-serif", size=13, color="#3B4945"),
        title_font=dict(size=17, color="#192522"),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#192522", bordercolor="#DDE5E1"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#DDE5E1", tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor="#E9EEEB", gridwidth=1, zeroline=False, tickfont=dict(size=12))
    st.plotly_chart(fig, width="stretch")


df = get_data()
filtered = filter_data(df)
pages = ["Overview", "User Engagement", "Conversion Funnel", "Retention", "Churn Analysis", "Churn Prediction", "Product Recommendations"]
requested_page = st.query_params.get("page", "Overview")
default_page = pages.index(requested_page) if requested_page in pages else 0
page = st.sidebar.radio("Analysis", pages, index=default_page)
if page != requested_page:
    st.query_params["page"] = page

st.markdown('<p class="hero-kicker">Product analytics case study</p>', unsafe_allow_html=True)
st.title("AI Product Analytics Dashboard")
st.markdown('<p class="hero-subtitle">Activation, engagement, retention, conversion, and churn insights for a simulated AI SaaS product.</p>', unsafe_allow_html=True)

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
    st.markdown(
        f'<div class="insight-line"><strong>What to notice:</strong> {metrics["most_used_feature"]} is the most-used capability in this selection. Compare acquisition volume with converted users, then review how revenue concentrates across plans.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 1])
    with left:
        monthly = filtered.assign(signup_month=filtered["signup_date"].dt.to_period("M").astype(str)).groupby("signup_month", as_index=False).agg(users=("user_id", "count"), paid=("trial_to_paid", "sum"))
        plot(px.line(monthly, x="signup_month", y=["users", "paid"], markers=True, title="Monthly acquisition and conversion"))
    with right:
        plan = filtered.groupby("plan_type", as_index=False).agg(users=("user_id", "count"), revenue=("monthly_revenue", "sum"))
        plot(px.bar(plan, x="plan_type", y="revenue", color="plan_type", title="Monthly revenue by plan"))

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
        st.markdown(f'<div class="recommendation"><h4>{rec["title"]}</h4><p><strong>Finding and evidence:</strong> {rec["finding"]}</p><p><strong>Proposed action:</strong> {rec["action"]}</p><p><strong>Expected impact:</strong> {rec["impact"]}</p></div>', unsafe_allow_html=True)
