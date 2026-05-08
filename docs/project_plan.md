# Credit Risk Scoring System — Project Plan

## Project Overview

A production-grade credit risk scoring system built on the Home Credit Default Risk dataset. The project goes beyond standard ML modeling to demonstrate a full production ML lifecycle: multi-table feature engineering, MLflow experiment governance, concept drift monitoring, and an LLM-powered explainability layer. Deployed via FastAPI (integrated with custom portfolio UI).

**Target audience for this document:** Personal reference during build. Not for public consumption until project is complete.

**Reference repo:** kozodoi/Kaggle_Home_Credit (feature engineering and LightGBM modeling logic only — all other layers are original work)

---

## Dataset

**Source:** Home Credit Default Risk (Kaggle)
**Size:** ~300,000 loan applications in the main table
**Target:** Binary — 0 (repaid), 1 (defaulted). ~8% default rate (severe class imbalance)
**Tables:** 7 total

| Table | Description | Rows |
|---|---|---|
| application_train/test | Main table. One row per loan application | ~300K |
| bureau | Previous credits at other institutions reported to Credit Bureau | ~1.7M |
| bureau_balance | Monthly balance snapshots of bureau credits | ~27M |
| previous_application | Previous Home Credit loan applications | ~1.7M |
| POS_CASH_balance | Monthly snapshots of POS and cash loans | ~10M |
| credit_card_balance | Monthly credit card balance snapshots | ~3.8M |
| installments_payments | Repayment history for previous loans | ~13.6M |

The core challenge: supplementary tables cannot be directly joined to the main table because one applicant has multiple rows in each. Every feature from these tables requires aggregation per applicant ID.

---

## Repository Structure

```
credit-risk-mlops/
│
├── data/
│   ├── raw/                    # Home Credit CSVs (gitignored)
│   └── processed/              # Aggregated feature dataset (gitignored)
│
├── src/
│   ├── features/
│   │   ├── application.py      # Features from main table
│   │   ├── bureau.py           # Features from bureau + bureau_balance
│   │   ├── previous_app.py     # Features from previous_application
│   │   ├── pos_cash.py         # Features from POS_CASH_balance
│   │   ├── credit_card.py      # Features from credit_card_balance
│   │   ├── installments.py     # Features from installments_payments
│   │   └── pipeline.py         # Orchestrates all feature modules
│   │
│   ├── models/
│   │   ├── train.py            # Training loop with MLflow logging
│   │   ├── evaluate.py         # Evaluation metrics and plots
│   │   ├── registry.py         # MLflow model registry operations
│   │   └── predict.py          # Inference logic
│   │
│   ├── monitoring/
│   │   ├── drift.py            # Evidently drift detection logic
│   │   ├── reports.py          # HTML report generation
│   │   └── triggers.py         # Drift threshold + retraining signal
│   │
│   └── explainer/
│       ├── shap_explainer.py   # SHAP value computation
│       └── llm_explainer.py    # Prompt engineering + Gemini API call
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis only
│   └── 02_feature_analysis.ipynb # Feature importance analysis
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── mlruns/                     # MLflow tracking directory
├── fastapi_app.py              # API serving layer
├── requirements.txt
├── .env.example                # API keys template
└── README.md
```

---

## Phases

### Phase 1 — Project Skeleton + MLflow Setup

**Goal:** Repo structure in place, MLflow running, plumbing verified end-to-end before any real modeling begins.

**What happens:**
- Initialize repo with the structure above
- Install dependencies: lightgbm, mlflow, scikit-learn, shap, evidently, fastapi
- Configure MLflow tracking server locally (SQLite backend, local artifact store)
- Write a smoke test: train a logistic regression on two columns from the main application table, log parameters and metrics to MLflow, register the model, promote it from None → Staging → Production
- Verify the MLflow UI shows the run, the registered model, and the artifacts correctly

**Done condition:** A dummy model exists in the MLflow registry at Production stage. MLflow UI is accessible at localhost:5000 and shows the run cleanly.

**Key decisions to make:**
- MLflow backend: local SQLite or remote (keep local for now, document how to switch)
- Artifact store: local filesystem (mlruns/) is fine for a portfolio project
- Model registry naming convention: establish this early and stick to it

---

### Phase 2 — EDA + Single Table Baseline

**Goal:** Understand the data deeply, establish a baseline AUC on the main table only.

**What happens:**

*EDA (notebook — not pipeline code):*
- Class imbalance: confirm ~8% default rate, document implications for metric choice
- Missing values: which features have >50% missing, decisions on imputation vs dropping
- Feature distributions: separate plots for defaulters vs non-defaulters on key features
- Correlations: identify multicollinearity candidates
- Target leakage audit: any features that could not be known at application time

*Baseline model (pipeline code):*
- Use application_train.csv only — no supplementary tables yet
- Temporal split: sort by SK_ID_CURR, use last 20% as test set (not random split)
- Handle class imbalance: test scale_pos_weight in LightGBM vs no handling, log both runs to MLflow
- Evaluate on: AUC-ROC (primary), AUC-PR (secondary, better for imbalanced problems), KS statistic
- Log everything to MLflow: parameters, all three metrics, feature importance plot as artifact

*Why AUC-PR matters here:*
AUC-ROC can look deceptively good on imbalanced data. With 8% defaults, a model that flags nothing scores 0.5 ROC but 0.08 PR. AUC-PR is the honest metric. Report both, use ROC as primary for comparability with kozodoi's work.

**Expected baseline AUC-ROC:** ~0.74–0.75 on main table only (consistent with published benchmarks)

**Done condition:** Baseline model logged to MLflow with AUC-ROC ~0.74+, temporal split in place, EDA notebook complete with documented findings.

---

### Phase 3 — Multi-Table Feature Engineering

**Goal:** Implement kozodoi's feature engineering logic as modular Python, add all supplementary tables one at a time, push AUC to 0.78–0.79 range.

**What happens per table:**

*Bureau + Bureau Balance:*
- Aggregate per SK_ID_CURR: count of bureau records, active vs closed accounts ratio, average days credit, max overdue days, credit utilization ratio
- From bureau_balance: proportion of months with overdue status, worst status ever recorded
- Key insight from kozodoi: time-windowed aggregations (last 12 months vs all history) add signal
- Retrain, log new run to MLflow, compare AUC delta vs previous run

*Previous Application:*
- Aggregate per SK_ID_CURR: approval rate on previous applications, average previous loan amount, average days between decision and disbursement
- Separate statistics for approved vs refused applications
- Most recent application features weighted more (recency matters)
- Retrain, log, compare

*POS_CASH Balance:*
- Aggregate per SK_ID_CURR: months past due statistics, completed vs active loans ratio, recent DPD trend
- Retrain, log, compare

*Credit Card Balance:*
- Aggregate per SK_ID_CURR: average credit utilization, payment ratio (amount paid / minimum required), balance trend over time
- Retrain, log, compare

*Installments Payments:*
- Aggregate per SK_ID_CURR: payment timeliness (days before/after due date), underpayment ratio, late payment frequency
- Retrain, log, compare

*Final model:*
- Full feature set (~150–300 features after all tables)
- Hyperparameter tuning: Optuna or manual grid, all runs logged to MLflow
- Probability calibration: Platt scaling or isotonic regression — the output needs to be a reliable probability, not just a score
- Final comparison: champion model selected, promoted to Production in MLflow registry

**Feature engineering philosophy:** Every aggregation should be explainable. If you can't say in one sentence why a feature captures credit risk signal, question whether it belongs.

**Expected final AUC-ROC:** 0.78–0.79

**Done condition:** Full feature pipeline in `src/features/`, final model at Production in MLflow registry with AUC-ROC ≥ 0.78, all experiment runs visible and comparable in MLflow UI.

---

### Phase 4 — Drift Monitoring

**Goal:** Simulate concept drift using temporal population split, detect it with Evidently AI, log drift events to MLflow.

**What happens:**

*Drift simulation setup:*
- Home Credit SK_ID_CURR correlates with application recency — higher IDs are more recent
- Reference population: SK_ID_CURR in lower range (training population — what the model was built on)
- Current population: SK_ID_CURR in higher range (simulates borrowers arriving post-deployment)
- These populations genuinely differ in feature distributions — this is real drift, not synthetic

*Evidently monitoring:*
- Run DataDriftPreset comparing reference vs current on all features
- Run DataQualityPreset to catch missing value pattern shifts
- Run ClassificationPreset if labels available — shows actual performance degradation alongside feature drift
- Generate HTML reports saved as MLflow artifacts

*Drift thresholds:*
- Flag dataset-level drift if >30% of features show statistically significant drift
- Flag critical drift if any of the top-10 most important SHAP features drift
- Document threshold reasoning — not arbitrary, tied to feature importance

*Retraining trigger:*
- When drift is flagged: log a drift_detected=True metric to MLflow, tag the current Production model as needs_review
- Simulate retraining on the newer population, compare AUC before and after
- This demonstrates the full monitoring → detection → retraining loop

**Done condition:** Drift report generated and saved as MLflow artifact. At least one drift event logged. Before/after AUC comparison showing model degradation and recovery after retraining on new population.

---

### Phase 5 — LLM Explanation Layer

**Goal:** For any prediction, generate a plain-English explanation of why the applicant was flagged as high or low risk.

**What happens:**

*SHAP computation:*
- Use TreeExplainer (appropriate for tree-based models, fast)
- For a given applicant, compute SHAP values for all features
- Extract top 5 features by absolute SHAP value — both positive (pushing toward default) and negative (pushing toward repayment)
- Format as structured context: feature name, actual value, SHAP contribution, direction

*Prompt engineering:*
- System prompt establishes context: credit risk analyst reviewing a loan application
- User prompt provides: applicant summary, risk score, top 5 SHAP contributors with values and directions
- Output instruction: 3–4 sentences, plain English, no jargon, explain the primary drivers
- Constraint: do not recommend approval or rejection — explain the risk factors only (regulatory boundary)

*Example prompt structure:*
```
You are a credit risk analyst. A loan application has been scored by an ML model.

Risk score: 0.73 (High Risk — above 0.5 threshold)

Top factors influencing this score:
- EXT_SOURCE_2: 0.31 (value: 0.18) — significantly increases default risk
- DAYS_EMPLOYED: 0.24 (value: -1200) — increases default risk
- AMT_CREDIT: -0.18 (value: 135000) — reduces default risk
- BUREAU_ACTIVE_CREDITS_COUNT: 0.15 (value: 8) — increases default risk
- PAYMENT_RATE: -0.12 (value: 0.043) — reduces default risk

Explain in 3-4 sentences why this applicant was flagged as high risk.
Do not recommend approval or rejection.
```

*Gemini API integration:*
- Use Gemini 1.5 Flash (fast, cheap, sufficient for structured output)
- Response parsed and returned as explanation field in API response
- Fallback: if API call fails, return SHAP summary in structured text format without LLM

*Validation:*
- Test on 10 sample applicants — high risk, low risk, borderline
- Verify explanations are coherent, accurate to SHAP values, and not hallucinated
- Document one or two examples in README

**Done condition:** `/predict` endpoint returns `{risk_score, top_shap_features, explanation}` for any applicant input. Explanations are coherent and consistent with SHAP values.

---

### Phase 6 — Deployment + Integration

**Goal:** Everything served via API and documented for integration.

**What happens:**

*FastAPI:*
- `POST /predict` — takes applicant JSON, runs feature pipeline, loads Production model from MLflow registry, computes SHAP, calls Gemini, returns full response
- `GET /health` — returns model version, registry stage, last drift check timestamp
- `GET /drift-report` — returns latest Evidently drift report URL or summary
- Input validation with Pydantic models
- Error handling: what happens if MLflow registry is unreachable, if Gemini API fails

*Documentation & Integration:*
- README: project overview, architecture diagram, setup instructions, example API call, sample output
- SHAP feature importance plot for the final model
- One complete prediction example end-to-end (input → score → explanation)
- Document API endpoints clearly for the custom portfolio UI to consume

**Done condition:** FastAPI server is running and accessible. A prediction request returns a risk score, SHAP chart data, and LLM explanation within 3 seconds. README is complete.

---

## Key Technical Decisions (Pre-committed)

| Decision | Choice | Reason |
|---|---|---|
| Dataset | Home Credit Default Risk | Multi-table complexity, well-documented, real lending problem |
| Primary metric | AUC-ROC | Standard for credit scoring, comparable to benchmarks |
| Secondary metric | AUC-PR | More honest on imbalanced data |
| Train/test split | Temporal (SK_ID_CURR order) | Mirrors production deployment, prevents leakage |
| Class imbalance | scale_pos_weight in LightGBM | Test vs alternatives in Phase 2, log both |
| MLflow backend | Local SQLite | Sufficient for portfolio, easy to demo |
| Drift tool | Evidently AI | Purpose-built for ML monitoring, clean HTML reports |
| Drift simulation | SK_ID_CURR temporal split | Real distributional differences, not synthetic |
| LLM | Gemini 1.5 Flash | Fast, cheap, API accessible |
| SHAP explainer | TreeExplainer | Exact values for tree models, fast |
| Deployment | FastAPI | Standard production stack, served to custom UI |

---

## What This Project Is NOT Doing

- No ensemble modeling (competition artifact, not production-relevant)
- No GRU/neural network features (overkill, not explainable)
- No real-time data pipeline (simulated drift is sufficient)
- No cloud deployment (local API serving is enough for portfolio)
- No hyperparameter search beyond Optuna basic (AUC target is 0.78, not 0.80)
- No automated retraining pipeline (trigger signal is sufficient, full automation is MLOps engineer scope)

---

## Resume Bullet (Target)

Built an end-to-end credit risk scoring system on 300K+ Home Credit loan applications — multi-table feature engineering across 7 data sources, full MLflow experiment tracking and model registry governance with staging/production lifecycle, Evidently AI drift monitoring simulating borrower population shift with retraining triggers, and a SHAP-to-LLM explanation layer generating plain-English credit decision rationale — served via FastAPI for integration with a custom portfolio interface.

---

## Open Questions (Resolve During Build)

- Does probability calibration meaningfully improve AUC-PR? Test in Phase 3.
- Which features from kozodoi's notebook are most critical vs safe to skip? Assess in Phase 3.
- Does Gemini explanation quality degrade on borderline predictions (score ~0.5)? Test in Phase 5.
- Should the custom UI show all features or a curated subset for the input form? Decide in Phase 6.
