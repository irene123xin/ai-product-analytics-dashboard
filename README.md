# AI Product Analytics Dashboard

An end-to-end product analytics case study for a simulated AI SaaS product, analyzing 7,500 users to understand activation, engagement, conversion, retention, and churn, and translating behavioral data into actionable product decisions.

**Live Demo:** [Open the dashboard](https://ai-appuct-analytics-dashboard-5edbb6rrzzefm9fvh9ed6y.streamlit.app/~/+/#ai-product-analytics-dashboard)

> This project uses a reproducible synthetic dataset designed to simulate realistic SaaS user behavior. The analysis demonstrates product analytics methodology and decision-making rather than production or causal claims.

## Product findings at a glance

- **Onboarding is an activation lever:** completed users converted to paid at 68.0%, compared with 49.8% for users who did not complete onboarding.
- **AI adoption marks materially deeper engagement:** adopters averaged 7.7 sessions and 7.1 active days in 30 days; non-adopters averaged 1.8 sessions and 1.7 active days.
- **Early inactivity is a clear risk signal:** users active on no more than two days churned at 37.3%, versus 10.6% for users active at least eight days.
- **Acquisition quality varies:** referral users had the highest paid conversion rate at 68.8% and the lowest churn rate at 16.2%.
- **The churn model is directionally useful, not production-ready:** Logistic Regression achieved 0.701 held-out ROC-AUC and outperformed Random Forest in this benchmark.

## Decisions supported by the analysis

- Instrument onboarding steps and test a shorter path to the first successful AI task.
- Trigger contextual activation support when a user has not tried the AI workflow within three sessions.
- Create a low-activity lifecycle segment for targeted education before users become inactive.
- Evaluate customer acquisition cost before testing increased referral investment.
- Connect repeated support issues to proactive follow-up and product reliability prioritization.

**Tech stack:** Python · pandas · NumPy · scikit-learn · Plotly · Streamlit · pytest · Jupyter

## Business problem

The product team needs to understand whether users reach value, where activation and conversion break down, which behaviors are associated with retention, and which segments warrant intervention. The project turns a user-level product snapshot into a concise set of decisions for product, growth, and operations teams.

## Key questions

- Which plans and acquisition channels bring engaged users?
- Where do users leave the activation and paid-conversion funnel?
- How does retention change across signup cohorts?
- Which behaviors distinguish churned and retained users?
- Can simple, explainable models identify churn risk better than chance?
- Which interventions are sufficiently supported to justify an experiment?

## Dataset

`data/product_users.csv` contains 7,500 deterministic users with signup and activity dates, plan and channel attributes, 30-day engagement, feature usage, onboarding, conversion, subscription status, churn, support demand, and monthly revenue.

The generator encodes internally consistent relationships rather than sampling every field independently. Onboarding and engagement influence product adoption, active paid plans determine revenue, and low engagement, incomplete onboarding, and support demand influence churn risk.

## Analysis workflow

1. Generate and validate a reproducible user-level snapshot.
2. Calculate executive, engagement, feature-adoption, and revenue metrics.
3. Construct a sequential activation and conversion funnel.
4. Estimate Month 0 through Month 3 cohort retention from signup and last activity.
5. Compare churned and retained users across behavior and segments.
6. Train Logistic Regression and Random Forest baselines on a stratified holdout split.
7. Translate descriptive and predictive signals into testable product recommendations.

## Key findings

- **Activation:** 71.8% completed onboarding, 61.9% used the AI feature, 44.1% converted their trial, and 27.6% became active paid users. This identifies paid activation, rather than initial feature discovery, as the largest remaining funnel opportunity.
- **Onboarding:** paid conversion reached 68.0% among onboarding completers versus 49.8% among non-completers. Onboarding quality should be evaluated as a commercial as well as usability metric.
- **Feature adoption:** AI adopters averaged 7.7 sessions and 7.1 active days, while non-adopters averaged 1.8 sessions and 1.7 active days. Early AI use is a practical activation milestone to instrument and test.
- **Churn:** overall churn was 21.2%. Onboarding completers churned at 16.9% versus 32.2% for non-completers, and AI adopters churned at 18.5% versus 33.6% for non-adopters.
- **Lifecycle risk:** users active on two days or fewer churned at 37.3%, compared with 10.6% among users active at least eight days. Active-day frequency offers a simple rule for identifying users who may benefit from intervention.
- **Acquisition:** referral users combined 68.8% paid conversion with 16.2% churn, the strongest observed channel profile. Budget decisions still require acquisition-cost and volume data.

## Product recommendations

| Finding | Evidence | Proposed action | Expected impact |
|---|---|---|---|
| Onboarding is associated with activation and retention | 68.0% vs. 49.8% paid conversion; 16.9% vs. 32.2% churn | Instrument step-level exits and A/B test a shorter first-run checklist | More users reaching activation and fewer early exits |
| AI adoption marks stronger engagement | 7.7 vs. 1.8 sessions; 18.5% vs. 33.6% churn | Test an example-led first task and a prompt for users without AI use after three sessions | Faster time to value and stronger early habits |
| Low active-day frequency identifies lifecycle risk | 37.3% churn at two days or fewer vs. 10.6% at eight days or more | Create a low-activity segment and test contextual use-case education | Earlier, more targeted retention intervention |
| Referral brought higher-quality users | 68.8% conversion and 16.2% churn | Add acquisition cost and volume, then run a bounded referral investment test | Better quality-adjusted acquisition decisions |
| Repeated support demand adds risk | Users with at least two tickets showed 24.9% churn | Tag recurring issue categories and route repeat contacts to proactive follow-up | Reduced friction and clearer reliability priorities |

These recommendations are hypotheses for experimentation. Observational associations do not establish causality.

## Machine learning approach

The classification workflow uses a 75/25 stratified train/test split and a shared preprocessing pipeline for numerical scaling, missing-value handling, and categorical one-hot encoding. Logistic Regression and Random Forest are evaluated with accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.

Class weighting addresses the churn imbalance. Logistic Regression produced the stronger held-out ROC-AUC at 0.701, compared with 0.673 for Random Forest. The dashboard reports these as offline baseline results, not operational forecasts.

## What This Project Demonstrates

- Product analytics and user behavior analysis
- Activation and conversion funnel analysis
- Signup-cohort retention analysis
- Churn modeling and machine-learning evaluation
- Translating behavioral evidence into product decisions
- Clear business communication for product and growth teams
- Streamlit dashboard development
- Reproducible Python data generation, validation, and analysis

## Repository structure

```text
ai-product-analytics-dashboard/
├── app.py
├── data/product_users.csv
├── notebooks/product_analysis.ipynb
├── src/
│   ├── analysis.py
│   ├── data_generation.py
│   └── model.py
├── tests/test_analytics.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.data_generation
pytest -q
streamlit run app.py
```

Open `http://localhost:8501`. To explore the notebook, run `jupyter lab notebooks/product_analysis.ipynb`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the GitHub account that owns the repository.
4. Select this repository and branch.
5. Set the main file path to `app.py`.
6. Select **Deploy**.

Streamlit installs `requirements.txt` automatically. The application uses repository-relative data paths and requires no secrets.

## Limitations

- Retention is estimated from a user-level snapshot rather than event-level activity.
- Churn models are evaluated offline and are not calibrated for operational intervention.
- Feature importance and segment differences are associative, not causal.
- Recommendations require product instrumentation, experiments, and customer research before implementation.

## Author

Yi Xin Ding
