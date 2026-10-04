"""Product analytics metrics used by the dashboard and notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_users(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signup_date", "last_active_date"])
    return df


def executive_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    feature_totals = {
        "AI generation": df["ai_feature_usage"].sum(),
        "Export": df["export_feature_usage"].sum(),
        "Collaboration": df["collaboration_feature_usage"].sum(),
    }
    return {
        "total_users": len(df),
        "active_users": int((df["active_days_30d"] > 0).sum()),
        "paid_conversion_rate": float(df["trial_to_paid"].mean()),
        "churn_rate": float(df["churned"].mean()),
        "arpu": float(df["monthly_revenue"].mean()),
        "most_used_feature": max(feature_totals, key=feature_totals.get),
    }


def conversion_funnel(df: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("Signed up", pd.Series(True, index=df.index)),
        ("Completed onboarding", df["onboarding_completed"].eq(1)),
        ("Used AI feature", df["onboarding_completed"].eq(1) & df["ai_feature_usage"].gt(0)),
        ("Converted trial", df["onboarding_completed"].eq(1) & df["ai_feature_usage"].gt(0) & df["trial_to_paid"].eq(1)),
        ("Active paid", df["onboarding_completed"].eq(1) & df["ai_feature_usage"].gt(0) & df["trial_to_paid"].eq(1) & df["subscription_status"].eq("Active Paid")),
    ]
    counts = [int(mask.sum()) for _, mask in stages]
    return pd.DataFrame(
        {
            "stage": [name for name, _ in stages],
            "users": counts,
            "overall_conversion": np.array(counts) / max(counts[0], 1),
            "step_conversion": [1.0] + [counts[i] / max(counts[i - 1], 1) for i in range(1, len(counts))],
        }
    )


def cohort_retention(df: pd.DataFrame, max_month: int = 3) -> pd.DataFrame:
    """Estimate retention from signup and last activity using user-level snapshot dates."""
    work = df.copy()
    work["signup_date"] = pd.to_datetime(work["signup_date"])
    work["last_active_date"] = pd.to_datetime(work["last_active_date"])
    work["cohort_month"] = work["signup_date"].dt.to_period("M")
    observed_month = (work["last_active_date"].dt.year - work["signup_date"].dt.year) * 12 + (
        work["last_active_date"].dt.month - work["signup_date"].dt.month
    )
    rows = []
    for cohort, group in work.groupby("cohort_month"):
        eligible_age = (pd.Timestamp("2026-09-30").to_period("M") - cohort).n
        for month in range(max_month + 1):
            if eligible_age >= month:
                retained = (observed_month.loc[group.index] >= month).mean()
                rows.append({"cohort_month": str(cohort), "month": month, "retention": retained})
    result = pd.DataFrame(rows)
    return result.pivot(index="cohort_month", columns="month", values="retention").rename(columns=lambda x: f"Month {x}")


def churn_comparison(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["sessions_30d", "active_days_30d", "ai_feature_usage", "avg_session_minutes", "support_tickets", "onboarding_completed"]
    result = df.groupby("churned")[metrics].mean().T
    result.columns = ["Retained", "Churned"]
    result["Difference"] = result["Churned"] - result["Retained"]
    return result


def product_recommendations(df: pd.DataFrame) -> list[dict[str, str]]:
    onboard = df.groupby("onboarding_completed")["churned"].mean()
    ai_users = df.assign(ai_adopted=df["ai_feature_usage"] > 0).groupby("ai_adopted")["churned"].mean()
    low = df[df["active_days_30d"] <= 2]["churned"].mean()
    high = df[df["active_days_30d"] >= 8]["churned"].mean()
    channel = df.groupby("acquisition_channel").agg(churn=("churned", "mean"), conversion=("trial_to_paid", "mean"))
    best_channel = channel.sort_values(["churn", "conversion"], ascending=[True, False]).index[0]
    ticketed = df[df["support_tickets"] >= 2]["churned"].mean()
    return [
        {"title": "Improve onboarding completion", "finding": f"Churn is {onboard.get(0, 0):.1%} without onboarding versus {onboard.get(1, 0):.1%} after completion.", "action": "Instrument each onboarding step, identify the largest exit point, and A/B test a shorter first-run checklist with contextual guidance.", "impact": "More users reaching activation and fewer early lifecycle exits."},
        {"title": "Drive early AI feature adoption", "finding": f"Users who try the AI workflow churn at {ai_users.get(True, 0):.1%}, compared with {ai_users.get(False, 0):.1%} for non-adopters.", "action": "Add an example-led first task and test a lifecycle prompt for users who have not tried AI within three sessions.", "impact": "Faster time to value and stronger early product habits."},
        {"title": "Create a low-engagement intervention", "finding": f"Users active on two days or fewer churn at {low:.1%}; users active at least eight days churn at {high:.1%}.", "action": "Define a low-activity lifecycle segment and test use-case education before users become fully inactive.", "impact": "Earlier, more targeted intervention for disengaging accounts."},
        {"title": "Rebalance acquisition investment", "finding": f"{best_channel} combines the strongest observed conversion and retention profile.", "action": "Compare customer acquisition cost and volume by channel, then run a bounded budget-allocation test.", "impact": "Improved quality-adjusted acquisition without assuming volume will scale unchanged."},
        {"title": "Close the support-risk loop", "finding": f"Users with two or more tickets show {ticketed:.1%} churn in this snapshot.", "action": "Tag recurring issue categories and route repeated-ticket accounts to proactive product or success follow-up.", "impact": "Reduced friction and clearer evidence for prioritizing reliability work."},
    ]
