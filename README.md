 # 🛡️ CoverFlow

## Segment-Conditional Conformal Prediction for Auditable, Workload-Aware Fraud Triage

CoverFlow is a financial transaction fraud triage system that combines **XGBoost** with **Conformal Prediction** to provide uncertainty-aware fraud decisions.

Instead of forcing every transaction into a simple Fraud/Non-Fraud prediction, CoverFlow generates prediction sets and uses uncertainty to decide whether a transaction can be automatically resolved or should be routed to a human analyst.

## Features

- XGBoost-based fraud detection
- Class-Conditional Conformal Prediction
- Transaction Amount Segment calibration
- Device Type Segment calibration
- Approximately 95% target coverage
- Workload-aware fraud triage
- Batch CSV transaction analysis
- Interactive Streamlit dashboard
- Coverage and analyst workload monitoring

## Triage Logic

| Prediction Set | Decision |
|---|---|
| `{Non-Fraud}` | Auto Resolve |
| `{Fraud}` | Fraud Alert |
| `{Non-Fraud, Fraud}` | Analyst Review |

## Experimental Results

| Method | Coverage | Mean Set Size | Analyst Workload |
|---|---:|---:|---:|
| Class-Conditional | 95.19% | 1.296 | 29.65% |
| Amount Segment | 95.18% | 1.316 | 31.58% |
| Device Segment | 95.24% | 1.370 | 36.96% |

**XGBoost ROC-AUC:** 0.9475

## Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset.

The training data is not included in this repository because of its size. A small set of demo transactions is included for demonstrating the deployed application.

## Technology Stack

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Conformal Prediction
- Streamlit

## Project Structure

```text
CoverFlow/
├── app.py
├── demo_transactions.csv
├── requirements.txt
├── README.md
├── .gitignore
│
└── artifacts/
    ├── xgboost_model.json
    ├── encoder.pkl
    ├── imputer.pkl
    ├── feature_columns.pkl
    ├── categorical_cols.pkl
    ├── numeric_cols.pkl
    ├── qhat_class.pkl
    ├── amount_segment_qhat.pkl
    ├── amount_config.pkl
    ├── device_segment_qhat.pkl
    └── metrics.json