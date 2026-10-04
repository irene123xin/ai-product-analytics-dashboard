"""Generate a deterministic, internally consistent synthetic AI SaaS dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
REFERENCE_DATE = pd.Timestamp("2026-09-30")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_product_users(n_users: int = 7500, seed: int = SEED) -> pd.DataFrame:
    """Return synthetic user-level product data with plausible behavioral relationships."""
    rng = np.random.default_rng(seed)
    signup_date = REFERENCE_DATE - pd.to_timedelta(rng.integers(15, 730, n_users), unit="D")
    plan_type = rng.choice(["Free", "Pro", "Team"], n_users, p=[0.56, 0.31, 0.13])
    channel = rng.choice(
        ["Organic Search", "Paid Search", "Social", "Referral", "Content", "Partnership"],
        n_users,
        p=[0.25, 0.20, 0.17, 0.16, 0.14, 0.08],
    )
    country = rng.choice(
        ["United States", "United Kingdom", "Canada", "Germany", "France", "Australia", "Other"],
        n_users,
        p=[0.31, 0.15, 0.11, 0.10, 0.09, 0.08, 0.16],
    )

    channel_quality = pd.Series(channel).map(
        {"Referral": 0.70, "Content": 0.48, "Organic Search": 0.35, "Partnership": 0.30,
         "Paid Search": 0.05, "Social": -0.10}
    ).to_numpy()
    plan_signal = pd.Series(plan_type).map({"Free": -0.45, "Pro": 0.65, "Team": 1.05}).to_numpy()
    onboarding_probability = _sigmoid(0.65 + channel_quality + 0.35 * plan_signal)
    onboarding_completed = rng.binomial(1, onboarding_probability)

    latent_engagement = rng.normal(0, 0.85, n_users) + 0.75 * onboarding_completed + plan_signal + channel_quality
    sessions_30d = np.clip(rng.poisson(np.exp(1.65 + 0.30 * latent_engagement)), 0, 90)
    inactive = rng.random(n_users) < _sigmoid(-1.6 - latent_engagement)
    sessions_30d[inactive] = rng.choice([0, 1], inactive.sum(), p=[0.82, 0.18])
    active_days_30d = np.minimum(30, np.minimum(sessions_30d, rng.binomial(30, _sigmoid(-1.15 + 0.42 * latent_engagement))))
    avg_session_minutes = np.where(
        sessions_30d > 0,
        np.clip(rng.normal(12.5 + 2.4 * latent_engagement, 4.0), 2.0, 42.0),
        0.0,
    )

    ai_feature_usage = np.clip(rng.poisson(np.maximum(0.2, sessions_30d * _sigmoid(-0.5 + latent_engagement))), 0, 150)
    export_feature_usage = np.clip(rng.poisson(np.maximum(0.05, sessions_30d * _sigmoid(-2.0 + 0.65 * latent_engagement + 0.55 * plan_signal))), 0, 70)
    collaboration_feature_usage = np.clip(
        rng.poisson(np.maximum(0.03, sessions_30d * _sigmoid(-2.25 + 0.55 * latent_engagement + (plan_type == "Team") * 1.35))),
        0,
        70,
    )
    support_tickets = np.clip(rng.poisson(0.28 + 0.18 * (plan_type != "Free") + 0.22 * (sessions_30d > 18)), 0, 8)

    conversion_probability = _sigmoid(-1.75 + 0.90 * onboarding_completed + 0.035 * ai_feature_usage + channel_quality)
    trial_to_paid = rng.binomial(1, conversion_probability)
    trial_to_paid = np.where(plan_type == "Free", trial_to_paid, 1)

    churn_probability = _sigmoid(
        -0.55
        - 0.055 * sessions_30d
        - 0.075 * active_days_30d
        - 0.018 * ai_feature_usage
        - 0.52 * onboarding_completed
        + 0.30 * support_tickets
        + (plan_type == "Free") * 0.45
        - channel_quality * 0.25
        + rng.normal(0, 0.45, n_users)
    )
    churned = rng.binomial(1, np.clip(churn_probability, 0.02, 0.88))
    subscription_status = np.where(churned == 1, "Churned", np.where(plan_type == "Free", "Free", "Active Paid"))
    monthly_revenue = np.select(
        [(subscription_status == "Active Paid") & (plan_type == "Pro"),
         (subscription_status == "Active Paid") & (plan_type == "Team")],
        [29.0, 89.0],
        default=0.0,
    )

    days_since_active = np.where(
        churned == 1,
        rng.integers(16, 91, n_users),
        np.clip(rng.poisson(np.maximum(0.5, 8 - 1.15 * latent_engagement)), 0, 30),
    )
    last_active_date = np.maximum(signup_date.values, (REFERENCE_DATE - pd.to_timedelta(days_since_active, unit="D")).values)

    return pd.DataFrame(
        {
            "user_id": [f"USR-{i:06d}" for i in range(1, n_users + 1)],
            "signup_date": pd.to_datetime(signup_date).date,
            "last_active_date": pd.to_datetime(last_active_date).date,
            "plan_type": plan_type,
            "acquisition_channel": channel,
            "country": country,
            "sessions_30d": sessions_30d,
            "active_days_30d": active_days_30d,
            "ai_feature_usage": ai_feature_usage,
            "export_feature_usage": export_feature_usage,
            "collaboration_feature_usage": collaboration_feature_usage,
            "avg_session_minutes": np.round(avg_session_minutes, 1),
            "support_tickets": support_tickets,
            "onboarding_completed": onboarding_completed,
            "trial_to_paid": trial_to_paid,
            "subscription_status": subscription_status,
            "churned": churned,
            "monthly_revenue": monthly_revenue,
        }
    )


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ValueError when core consistency constraints are violated."""
    required = {"user_id", "signup_date", "last_active_date", "plan_type", "sessions_30d", "churned", "monthly_revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["user_id"].duplicated().any():
        raise ValueError("user_id must be unique")
    if not df["churned"].isin([0, 1]).all():
        raise ValueError("churned must be binary")
    if (df[["sessions_30d", "monthly_revenue"]].select_dtypes("number") < 0).any().any():
        raise ValueError("Usage and revenue values cannot be negative")
    if ((df["subscription_status"] != "Active Paid") & (df["monthly_revenue"] > 0)).any():
        raise ValueError("Only active paid users may have monthly revenue")


def save_dataset(output_path: str | Path, n_users: int = 7500, seed: int = SEED) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_product_users(n_users=n_users, seed=seed)
    validate_dataset(df)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data = save_dataset(project_root / "data" / "product_users.csv")
    print(f"Generated {len(data):,} synthetic users")
