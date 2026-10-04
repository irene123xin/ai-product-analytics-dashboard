from src.analysis import cohort_retention, conversion_funnel, executive_metrics
from src.data_generation import generate_product_users, validate_dataset
from src.model import train_churn_models


def test_generated_dataset_is_consistent():
    df = generate_product_users(500, seed=7)
    validate_dataset(df)
    assert df["user_id"].nunique() == 500
    assert ((df["monthly_revenue"] > 0) == (df["subscription_status"] == "Active Paid")).all()


def test_product_metrics_and_funnel():
    df = generate_product_users(500, seed=8)
    metrics = executive_metrics(df)
    funnel = conversion_funnel(df)
    assert metrics["total_users"] == 500
    assert funnel["users"].is_monotonic_decreasing
    assert cohort_retention(df).shape[1] == 4


def test_churn_models_return_expected_metrics():
    results = train_churn_models(generate_product_users(900, seed=9))
    assert set(results.metrics["Model"]) == {"Logistic Regression", "Random Forest"}
    assert results.metrics["ROC-AUC"].between(0, 1).all()
