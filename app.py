import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CoverFlow",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CoverFlow")
st.subheader("Uncertainty-Aware Financial Fraud Triage")

st.caption(
    "XGBoost + Segment-Conditional Conformal Prediction "
    "for auditable and workload-aware financial fraud monitoring"
)


# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():

    model = XGBClassifier()
    model.load_model("artifacts/xgboost_model.json")

    encoder = joblib.load("artifacts/encoder.pkl")
    imputer = joblib.load("artifacts/imputer.pkl")

    feature_columns = joblib.load(
        "artifacts/feature_columns.pkl"
    )

    categorical_cols = joblib.load(
        "artifacts/categorical_cols.pkl"
    )

    numeric_cols = joblib.load(
        "artifacts/numeric_cols.pkl"
    )

    qhat_class = joblib.load(
        "artifacts/qhat_class.pkl"
    )

    amount_qhat = joblib.load(
        "artifacts/amount_segment_qhat.pkl"
    )

    amount_config = joblib.load(
        "artifacts/amount_config.pkl"
    )

    device_qhat = joblib.load(
        "artifacts/device_segment_qhat.pkl"
    )

    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)

    return (
        model,
        encoder,
        imputer,
        feature_columns,
        categorical_cols,
        numeric_cols,
        qhat_class,
        amount_qhat,
        amount_config,
        device_qhat,
        metrics
    )


(
    model,
    encoder,
    imputer,
    feature_columns,
    categorical_cols,
    numeric_cols,
    qhat_class,
    amount_qhat,
    amount_config,
    device_qhat,
    metrics
) = load_artifacts()


# =========================================================
# LOAD DEMO TRANSACTIONS
# =========================================================
@st.cache_data
def load_demo_data():
    return pd.read_csv("demo_transactions.csv")


demo_df = load_demo_data()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_amount_segment(amount):

    if amount <= amount_config["q1"]:
        return "Low"

    elif amount <= amount_config["q2"]:
        return "Medium"

    return "High"


def preprocess_data(raw_df):

    X = raw_df.copy()

    if "TransactionID" in X.columns:
        X = X.drop(columns=["TransactionID"])

    if "isFraud" in X.columns:
        X = X.drop(columns=["isFraud"])

    # Add any missing features
    for col in feature_columns:

        if col not in X.columns:

            if col in categorical_cols:
                X[col] = "Unknown"
            else:
                X[col] = np.nan

    # Correct feature order
    X = X[feature_columns]

    # Categorical preprocessing
    X[categorical_cols] = (
        X[categorical_cols]
        .fillna("Unknown")
    )

    X[categorical_cols] = encoder.transform(
        X[categorical_cols]
    )

    # Numerical preprocessing
    X[numeric_cols] = imputer.transform(
        X[numeric_cols]
    )

    return X


def get_prediction_set(
    probabilities,
    mode,
    amount_segment,
    device
):

    prediction_set = []

    for c in [0, 1]:

        if mode == "Class-Conditional":

            qhat = qhat_class[c]

        elif mode == "Amount Segment":

            qhat = amount_qhat[
                (amount_segment, c)
            ]

        else:

            if device not in [
                "Unknown",
                "desktop",
                "mobile"
            ]:
                device = "Unknown"

            qhat = device_qhat[
                (device, c)
            ]

        if probabilities[c] >= (1 - qhat):

            if c == 0:
                prediction_set.append("Non-Fraud")
            else:
                prediction_set.append("Fraud")

    return prediction_set


def get_decision(prediction_set):

    if len(prediction_set) == 2:
        return "Analyst Review"

    if prediction_set == ["Fraud"]:
        return "Fraud Alert"

    if prediction_set == ["Non-Fraud"]:
        return "Auto Resolve"

    # Safety fallback
    return "Analyst Review"


def analyze_transactions(raw_df, mode):

    X = preprocess_data(raw_df)

    probabilities = model.predict_proba(X)

    results = []

    for i in range(len(raw_df)):

        row = raw_df.iloc[i]

        # Amount
        if "TransactionAmt" in raw_df.columns:
            amount = row["TransactionAmt"]
        else:
            amount = 0

        amount_segment = get_amount_segment(
            amount
        )

        # Device
        if "DeviceType" in raw_df.columns:

            device = row["DeviceType"]

            if pd.isna(device):
                device = "Unknown"

            device = str(device)

        else:
            device = "Unknown"

        fraud_probability = probabilities[i, 1]

        prediction_set = get_prediction_set(
            probabilities[i],
            mode,
            amount_segment,
            device
        )

        decision = get_decision(
            prediction_set
        )

        transaction_id = (
            row["TransactionID"]
            if "TransactionID" in raw_df.columns
            else i + 1
        )

        actual_label = None

        if "isFraud" in raw_df.columns:
            actual_label = int(
                row["isFraud"]
            )

        results.append(
            {
                "Transaction ID": transaction_id,
                "Amount": amount,
                "Amount Segment": amount_segment,
                "Device Type": device,
                "Fraud Probability": fraud_probability,
                "Prediction Set": prediction_set,
                "Decision": decision,
                "Actual Label": actual_label
            }
        )

    return results


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ CoverFlow Settings")

mode = st.sidebar.selectbox(
    "Calibration Mode",
    [
        "Class-Conditional",
        "Amount Segment",
        "Device Segment"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### Calibration Target

**95% Coverage**

Prediction sets represent model uncertainty.

Singleton sets can be automatically resolved,
while ambiguous sets are routed for analyst review.
"""
)


# =========================================================
# PERFORMANCE DASHBOARD
# =========================================================
st.markdown("## 📊 System Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "XGBoost ROC-AUC",
    f"{metrics['roc_auc']:.3f}"
)

c2.metric(
    "Class Coverage",
    f"{metrics['class_conditional_coverage'] * 100:.2f}%"
)

c3.metric(
    "Amount Coverage",
    f"{metrics['amount_segment_coverage'] * 100:.2f}%"
)

c4.metric(
    "Device Coverage",
    f"{metrics['device_segment_coverage'] * 100:.2f}%"
)

st.divider()


# =========================================================
# TRANSACTION ANALYSIS
# =========================================================
st.markdown("## 🔍 Transaction Triage")

input_mode = st.radio(
    "Choose transaction source",
    [
        "Demo Transactions",
        "Upload CSV"
    ],
    horizontal=True
)


# =========================================================
# DEMO MODE
# =========================================================
if input_mode == "Demo Transactions":

    st.info(
        "Select a real transaction sampled from the "
        "IEEE-CIS test split used during CoverFlow evaluation."
    )

    transaction_options = []

    for i in range(len(demo_df)):

        row = demo_df.iloc[i]

        transaction_id = row.get(
            "TransactionID",
            i + 1
        )

        actual = int(
            row["isFraud"]
        )

        label = (
            "Fraud"
            if actual == 1
            else "Non-Fraud"
        )

        transaction_options.append(
            f"Transaction {transaction_id} — Actual: {label}"
        )

    selected = st.selectbox(
        "Select Demo Transaction",
        range(len(transaction_options)),
        format_func=lambda x: transaction_options[x]
    )

    selected_df = demo_df.iloc[
        [selected]
    ].copy()

    selected_row = selected_df.iloc[0]

    # Basic transaction information
    st.markdown("### Transaction Information")

    t1, t2, t3, t4 = st.columns(4)

    transaction_id = selected_row.get(
        "TransactionID",
        selected + 1
    )

    amount = selected_row.get(
        "TransactionAmt",
        0
    )

    device = selected_row.get(
        "DeviceType",
        "Unknown"
    )

    if pd.isna(device):
        device = "Unknown"

    product = selected_row.get(
        "ProductCD",
        "Unknown"
    )

    t1.metric(
        "Transaction ID",
        str(transaction_id)
    )

    t2.metric(
        "Transaction Amount",
        f"{amount:.2f}"
    )

    t3.metric(
        "Device",
        str(device)
    )

    t4.metric(
        "Product",
        str(product)
    )

    if st.button(
        "🔎 Analyze Transaction",
        type="primary",
        use_container_width=True
    ):

        results = analyze_transactions(
            selected_df,
            mode
        )

        result = results[0]

        st.divider()

        st.markdown(
            "## CoverFlow Decision"
        )

        probability = result[
            "Fraud Probability"
        ]

        prediction_set = result[
            "Prediction Set"
        ]

        decision = result[
            "Decision"
        ]

        actual = result[
            "Actual Label"
        ]

        # ---------------------------------------------
        # Probability / Segment Information
        # ---------------------------------------------
        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        r2.metric(
            "Amount Segment",
            result["Amount Segment"]
        )

        r3.metric(
            "Device Segment",
            result["Device Type"]
        )

        actual_text = (
            "Fraud"
            if actual == 1
            else "Non-Fraud"
        )

        r4.metric(
            "Actual Label",
            actual_text
        )

        st.markdown(
            "### Conformal Prediction Set"
        )

        if len(prediction_set) == 2:

            st.warning(
                "Prediction Set: "
                "{ Non-Fraud, Fraud }"
            )

            st.caption(
                "The system cannot confidently "
                "exclude either class."
            )

        elif prediction_set == ["Fraud"]:

            st.error(
                "Prediction Set: { Fraud }"
            )

        elif prediction_set == ["Non-Fraud"]:

            st.success(
                "Prediction Set: { Non-Fraud }"
            )

        else:

            st.warning(
                "Prediction Set is empty. "
                "Transaction is routed for review."
            )

        st.markdown(
            "### Final Triage Decision"
        )

        if decision == "Auto Resolve":

            st.success(
                "✅ AUTO RESOLVE — "
                "Transaction classified as confident non-fraud."
            )

        elif decision == "Fraud Alert":

            st.error(
                "🚨 FRAUD ALERT — "
                "Transaction requires fraud investigation."
            )

        else:

            st.warning(
                "⚠️ ANALYST REVIEW — "
                "Model uncertainty requires human review."
            )

        # Compare actual outcome
        if (
            actual == 1 and
            "Fraud" in prediction_set
        ) or (
            actual == 0 and
            "Non-Fraud" in prediction_set
        ):

            st.caption(
                "✓ Actual class is contained in the "
                "conformal prediction set."
            )

        else:

            st.caption(
                "Actual class is not contained in the "
                "prediction set (miscoverage case)."
            )


# =========================================================
# CSV UPLOAD MODE
# =========================================================
else:

    st.info(
        "Upload IEEE-CIS formatted transaction data "
        "for batch fraud triage."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        raw_df = pd.read_csv(
            uploaded_file
        )

        st.markdown(
            "### Uploaded Transactions"
        )

        st.dataframe(
            raw_df.head(10),
            use_container_width=True
        )

        st.caption(
            f"{len(raw_df):,} transactions loaded."
        )

        if st.button(
            "🚀 Analyze Transactions",
            type="primary",
            use_container_width=True
        ):

            try:

                results = analyze_transactions(
                    raw_df,
                    mode
                )

                display_results = []

                for result in results:

                    prediction_set = result[
                        "Prediction Set"
                    ]

                    if len(prediction_set) == 2:
                        set_text = "{Non-Fraud, Fraud}"

                    elif len(prediction_set) == 1:
                        set_text = (
                            "{"
                            + prediction_set[0]
                            + "}"
                        )

                    else:
                        set_text = "{}"

                    display_results.append(
                        {
                            "Transaction ID":
                                result["Transaction ID"],

                            "Fraud Probability":
                                f"{result['Fraud Probability'] * 100:.2f}%",

                            "Amount Segment":
                                result["Amount Segment"],

                            "Device Type":
                                result["Device Type"],

                            "Prediction Set":
                                set_text,

                            "Decision":
                                result["Decision"]
                        }
                    )

                result_df = pd.DataFrame(
                    display_results
                )

                st.success(
                    "Transaction analysis completed."
                )

                st.markdown(
                    "### CoverFlow Triage Results"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                # -----------------------------------------
                # TRIAGE SUMMARY
                # -----------------------------------------
                auto_count = (
                    result_df["Decision"]
                    == "Auto Resolve"
                ).sum()

                review_count = (
                    result_df["Decision"]
                    == "Analyst Review"
                ).sum()

                fraud_count = (
                    result_df["Decision"]
                    == "Fraud Alert"
                ).sum()

                total = len(result_df)

                st.markdown(
                    "### Triage Summary"
                )

                s1, s2, s3, s4 = st.columns(4)

                s1.metric(
                    "Transactions",
                    total
                )

                s2.metric(
                    "Auto Resolved",
                    auto_count
                )

                s3.metric(
                    "Analyst Review",
                    review_count
                )

                s4.metric(
                    "Fraud Alerts",
                    fraud_count
                )

                if total > 0:

                    workload = (
                        review_count /
                        total
                    ) * 100

                    st.metric(
                        "Current Batch Analyst Workload",
                        f"{workload:.2f}%"
                    )

            except Exception as e:

                st.error(
                    "Transaction analysis failed."
                )

                st.exception(e)


# =========================================================
# OPERATIONAL MONITORING
# =========================================================
st.divider()

st.markdown(
    "## 📈 Coverage & Workload Monitoring"
)

comparison = pd.DataFrame(
    {
        "Calibration Method": [
            "Class Conditional",
            "Amount Segment",
            "Device Segment"
        ],

        "Coverage (%)": [
            metrics[
                "class_conditional_coverage"
            ] * 100,

            metrics[
                "amount_segment_coverage"
            ] * 100,

            metrics[
                "device_segment_coverage"
            ] * 100
        ],

        "Mean Set Size": [
            metrics[
                "class_conditional_mean_set_size"
            ],

            metrics[
                "amount_segment_mean_set_size"
            ],

            metrics[
                "device_segment_mean_set_size"
            ]
        ],

        "Analyst Workload (%)": [
            metrics[
                "class_conditional_workload"
            ],

            metrics[
                "amount_segment_workload"
            ],

            metrics[
                "device_segment_workload"
            ]
        ]
    }
)

st.dataframe(
    comparison.style.format(
        {
            "Coverage (%)": "{:.2f}",
            "Mean Set Size": "{:.3f}",
            "Analyst Workload (%)": "{:.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# WORKLOAD CHART
# =========================================================
st.markdown(
    "### Analyst Workload by Calibration Method"
)

workload_chart = (
    comparison[
        [
            "Calibration Method",
            "Analyst Workload (%)"
        ]
    ]
    .set_index(
        "Calibration Method"
    )
)

st.bar_chart(
    workload_chart
)


# =========================================================
# COVERAGE CHART
# =========================================================
st.markdown(
    "### Coverage by Calibration Method"
)

coverage_chart = (
    comparison[
        [
            "Calibration Method",
            "Coverage (%)"
        ]
    ]
    .set_index(
        "Calibration Method"
    )
)

st.bar_chart(
    coverage_chart
)


# =========================================================
# PROJECT EXPLANATION
# =========================================================
st.divider()

with st.expander(
    "ℹ️ How CoverFlow Works"
):

    st.markdown(
        """
### 1. Fraud Detection

An **XGBoost classifier** estimates the probability
that a financial transaction is fraudulent.

### 2. Conformal Calibration

CoverFlow applies **conformal prediction** to convert
model probabilities into prediction sets with a target
coverage level of approximately **95%**.

### 3. Segment-Conditional Calibration

Coverage can be evaluated and calibrated across
operationally meaningful groups such as:

- Transaction amount bands
- Device type

This helps expose uncertainty that may be hidden by
aggregate metrics.

### 4. Workload-Aware Triage

CoverFlow converts prediction-set uncertainty into
operational decisions:

**{Non-Fraud} → Auto Resolve**

**{Fraud} → Fraud Alert**

**{Non-Fraud, Fraud} → Analyst Review**

This allows the system to balance predictive
uncertainty with human analyst workload.
"""
    )


st.caption(
    "CoverFlow — Segment-Conditional Conformal Prediction "
    "for Auditable, Workload-Aware Fraud Triage"
)