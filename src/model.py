"""Train and evaluate reproducible churn classification baselines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["sessions_30d", "active_days_30d", "ai_feature_usage", "export_feature_usage", "collaboration_feature_usage", "avg_session_minutes", "support_tickets", "onboarding_completed", "trial_to_paid"]
CATEGORICAL_FEATURES = ["plan_type", "acquisition_channel", "country"]


@dataclass
class ModelResults:
    metrics: pd.DataFrame
    confusion_matrices: dict[str, list[list[int]]]
    feature_importance: pd.DataFrame
    best_model_name: str


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ]
    )


def train_churn_models(df: pd.DataFrame, random_state: int = 42) -> ModelResults:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced", random_state=random_state, n_jobs=-1),
    }
    rows, matrices, trained = [], {}, {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", _preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, prob)})
        matrices[name] = confusion_matrix(y_test, pred).tolist()
        trained[name] = pipeline
    metrics = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = metrics.loc[0, "Model"]
    best = trained[best_name]
    feature_names = best.named_steps["preprocess"].get_feature_names_out()
    model = best.named_steps["model"]
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0])
    importance = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).head(15)
    importance["feature"] = importance["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    return ModelResults(metrics, matrices, importance, best_name)
