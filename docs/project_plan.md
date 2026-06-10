# Credit Risk Scoring System — Fast-Track Project Plan

## Project Overview

A resume-focused, production-grade credit risk scoring system built on the Home Credit Default Risk dataset. Instead of building from scratch, this project uses existing legacy code as a launchpad, focusing on adding MLOps best practices, monitoring, and an LLM layer to create a high-impact portfolio piece quickly.

**Key Goals:**
- Use legacy data prep and modeling as a baseline.
- Handle large data constraints via ID-based sampling and university hardware (4090).
- Add MLflow for experiment tracking (starts after data prep).
- Add Evidently AI for drift monitoring.
- Add an LLM explanation layer (Gemini) with RAG for policy grounding and actionable advice (No SHAP).
- (Optional) Port feature engineering from Pandas to Polars for resume value.

---

## Dataset

**Source:** Home Credit Default Risk (Kaggle)
**Size:** ~300,000 loan applications in the main table.
**Target:** Binary — 0 (repaid), 1 (defaulted).
**Tables:** 7 total (Relational/Hierarchical).

| Table | Description | Rows |
|---|---|---|
| application_train/test | Main table. One row per loan application | ~300K |
| bureau | Previous credits at other institutions reported to Credit Bureau | ~1.7M |
| bureau_balance | Monthly balance snapshots of bureau credits | ~27M |
| previous_application | Previous Home Credit loan applications | ~1.7M |
| POS_CASH_balance | Monthly snapshots of POS and cash loans | ~10M |
| credit_card_balance | Monthly credit card balance snapshots | ~3.8M |
| installments_payments | Repayment history for previous loans | ~13.6M |

---

## Phases

### Phase 1 — Data Prep & Aggregation (Legacy Launchpad)

**Goal:** Process the multi-table dataset into a single flat file with aggregated features.

**Approach:**
- Use `legacy_code/code_1_data_prep.ipynb` as the base.
- **Local Dev (Today):** Address memory limits by sampling ~50k `SK_ID_CURR` from `application_train` and filtering all other tables to match.
- **Full Run (Tomorrow):** Run the full script on the university 4090 machine (check RAM availability).
- **Resume Flex:** Consider porting the Pandas operations to Polars for speed and modern stack representation.
- **Output:** A single Parquet file containing aggregated features for all applicants.

---

### Phase 2 — Modeling & MLflow Setup

**Goal:** Train the model and track experiments.

**Approach:**
- Use `legacy_code/code_2_modeling.ipynb` as the base.
- **Timing:** This phase starts *after* the data prep is completed and the aggregated file is ready.
- **MLflow:** Integrate MLflow tracking to log parameters, metrics (AUC-ROC, AUC-PR), and model artifacts.
- **Output:** A trained LightGBM model registered in the MLflow Model Registry.

---

### Phase 3 — LLM Explanation & RAG Layer (No SHAP)

**Goal:** Generate plain-English explanations for credit decisions grounded in company policy and provide actionable advice.

**Approach:**
- **No SHAP:** Do not use SHAP values. Use LightGBM's built-in feature importance to identify top predictive features.
- **RAG Guidebook:** Create a `credit_policy_guide.txt` containing dummy bank policies (e.g., thresholds for `EXT_SOURCE` or `DAYS_EMPLOYED`) and actionable advice for rejected applicants.
- **Retrieval-Augmented Generation (RAG):** When generating an explanation, the system retrieves relevant rules/advice from the guidebook based on the applicant's top features.
- **LLM Output:** Gemini 1.5 Flash generates a response that explains the decision *and* provides specific steps the applicant can take to improve their score (e.g., "Reduce loan amount", "Wait for longer employment history").

---

### Phase 4 — Drift Monitoring

**Goal:** Demonstrate ML monitoring capabilities.

**Approach:**
- Use **Evidently AI** to detect data drift.
- Simulate drift by splitting the data by `SK_ID_CURR` (proxy for time/recency).
- Generate HTML reports showing feature distribution shifts.

---

### Phase 5 — Deployment (FastAPI)

**Goal:** Serve the model and explanations via API.

**Approach:**
- Create a FastAPI app with a `/predict` endpoint.
- Endpoint takes applicant data, loads the model from MLflow, gets feature importance, calls Gemini with RAG, and returns the score + explanation + advice.

---

## Tech Stack (Resume Focus)
- **Data:** Polars (or Pandas), Parquet.
- **ML:** LightGBM.
- **MLOps:** MLflow, Evidently AI.
- **GenAI:** Gemini 1.5 Flash (via API), RAG (Custom implementation).
- **Deployment:** FastAPI.
