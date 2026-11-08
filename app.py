from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "FraudFound_P"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree (Entropy)": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(
    page_title="ClaimLens | Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the interface theme-aware. Avoid forcing light metric cards in dark mode.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1440px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .claimlens-hero {
            padding: 1.35rem 1.6rem;
            border-radius: 16px;
            background: linear-gradient(115deg, #10243a 0%, #1d4967 100%);
            color: #ffffff;
            margin-bottom: 1.1rem;
        }
        .claimlens-hero h1 {
            color: #ffffff !important;
            margin: 0;
            font-size: 2rem;
            line-height: 1.2;
        }
        .claimlens-hero p {
            color: #e4edf5 !important;
            margin: .45rem 0 0 0;
            font-size: 1rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, .28);
            border-radius: 12px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] p {
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-weight: 700;
        }
        .section-note {
            opacity: .78;
            font-size: .9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="claimlens-hero">
        <h1>ClaimLens | Vehicle Claim Fraud Analytics</h1>
        <p>Model evaluation and claim-level screening using labelled vehicle-insurance records.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "A fraud prediction is a screening signal for further review. "
    "It is not proof of fraud and must not be used on its own to reject a claim."
)

with st.sidebar:
    st.header("Evaluation setup")
    st.caption("Choose a trained classifier and upload a labelled evaluation CSV.")
    st.markdown("---")
    st.markdown("**Target variable**")
    st.code("FraudFound_P")
    st.caption("0 = Non-fraud  |  1 = Fraud")
    st.markdown("---")
    st.caption("All six evaluation metrics are calculated against the actual target labels.")

# Use the repository's existing `models/` directory, matching the training export.
model_path_for = lambda label: ROOT / "models" / MODEL_FILES[label]


@st.cache_resource(show_spinner="Loading trained model...")
def load_pipeline(path_string: str):
    return joblib.load(path_string)


st.subheader("1. Configure evaluation")

with st.form("evaluation_form", clear_on_submit=False):
    form_left, form_right = st.columns([1, 2], vertical_alignment="bottom")
    with form_left:
        selected_model = st.selectbox(
            "Classification model",
            list(MODEL_FILES.keys()),
            index=list(MODEL_FILES.keys()).index("SVM"),
        )
    with form_right:
        uploaded_file = st.file_uploader(
            "Upload labelled test data (CSV)",
            type=["csv"],
            help=(
                "The CSV must contain all model input columns and the actual "
                "FraudFound_P target column."
            ),
        )
    submitted = st.form_submit_button(
        "Run evaluation",
        type="primary",
        use_container_width=True,
    )

if not submitted:
    st.info(
        "Select a model, upload `test_data.csv`, then click **Run evaluation**. "
        "Results will not be calculated until you submit the form."
    )
    st.stop()

if uploaded_file is None:
    st.error("Please upload a labelled CSV file before running the evaluation.")
    st.stop()

model_path = model_path_for(selected_model)
if not model_path.exists():
    st.error(
        f"Trained model file was not found: `{model_path.relative_to(ROOT)}`. "
        "Confirm that the model files are present in the `models/` folder."
    )
    st.stop()

try:
    evaluation_data = pd.read_csv(uploaded_file)

    if evaluation_data.empty:
        st.error("The uploaded CSV contains no records.")
        st.stop()

    if TARGET not in evaluation_data.columns:
        st.error(
            f"The uploaded CSV must include the actual target column `{TARGET}` "
            "to calculate the six evaluation metrics."
        )
        st.stop()

    actual_numeric = pd.to_numeric(evaluation_data[TARGET], errors="coerce")
    if actual_numeric.isna().any():
        st.error(f"`{TARGET}` contains missing or non-numeric values.")
        st.stop()

    actual = actual_numeric.astype(int)
    if not set(actual.unique()).issubset({0, 1}):
        st.error(f"`{TARGET}` must contain only 0 (non-fraud) and 1 (fraud).")
        st.stop()

    if actual.nunique() != 2:
        st.error(
            "The uploaded evaluation file must contain both target classes "
            "(0 and 1) to calculate ROC-AUC."
        )
        st.stop()

    pipeline = load_pipeline(str(model_path))
    expected_features = list(pipeline.feature_names_in_)
    missing_features = sorted(set(expected_features) - set(evaluation_data.columns))

    if missing_features:
        st.error(
            "The uploaded CSV is missing required model features: "
            + ", ".join(missing_features)
        )
        st.stop()

    features = evaluation_data.loc[:, expected_features]
    predictions = pipeline.predict(features).astype(int)

    if not hasattr(pipeline, "predict_proba"):
        st.error("The selected model does not provide fraud probabilities for AUC.")
        st.stop()

    fraud_probability = pipeline.predict_proba(features)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(actual, predictions),
        "AUC Score": roc_auc_score(actual, fraud_probability),
        "Precision": precision_score(actual, predictions, zero_division=0),
        "Recall": recall_score(actual, predictions, zero_division=0),
        "F1 Score": f1_score(actual, predictions, zero_division=0),
        "MCC Score": matthews_corrcoef(actual, predictions),
    }

    st.divider()
    st.subheader("2. Evaluation results")
    st.caption(
        f"Selected model: **{selected_model}**  ·  "
        f"Evaluation records: **{len(evaluation_data):,}**"
    )

    metric_columns = st.columns(3)
    metric_order = [
        ("Accuracy", "Accuracy"),
        ("AUC Score", "ROC-AUC"),
        ("MCC Score", "MCC"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1 Score", "F1 Score"),
    ]
    for column, (metric_key, display_label) in zip(
        [metric_columns[0], metric_columns[1], metric_columns[2],
         *st.columns(3)],
        metric_order,
    ):
        column.metric(display_label, f"{metrics[metric_key]:.4f}")

    matrix = confusion_matrix(actual, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    st.subheader("3. Confusion matrix")
    matrix_left, matrix_right = st.columns([1.1, 1])
    with matrix_left:
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual non-fraud", "Actual fraud"],
                columns=["Predicted non-fraud", "Predicted fraud"],
            ),
            use_container_width=True,
        )
    with matrix_right:
        st.markdown("**Claim screening counts**")
        count_columns = st.columns(2)
        count_columns[0].metric("Correctly flagged fraud", f"{tp:,}")
        count_columns[1].metric("False fraud alerts", f"{fp:,}")
        count_columns = st.columns(2)
        count_columns[0].metric("Missed fraud cases", f"{fn:,}")
        count_columns[1].metric("Correctly cleared non-fraud", f"{tn:,}")

    report = classification_report(
        actual,
        predictions,
        labels=[0, 1],
        target_names=["Non-fraud", "Fraud"],
        output_dict=True,
        zero_division=0,
    )
    with st.expander("View detailed classification report"):
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    # Preserve the uploaded row order and all original columns for traceability.
    results = evaluation_data.copy()
    results.insert(0, "Evaluation_Row", range(1, len(results) + 1))
    results["Predicted_FraudFound_P"] = predictions
    results["Fraud_Probability"] = fraud_probability
    results["Prediction_Status"] = pd.Series(predictions).map(
        {0: "Predicted non-fraud", 1: "Flagged for fraud review"}
    ).values
    results["Actual_Outcome"] = actual.map({0: "Actual non-fraud", 1: "Actual fraud"}).values

    flagged = results.loc[results["Predicted_FraudFound_P"] == 1].copy()
    correctly_flagged = flagged.loc[flagged[TARGET] == 1].copy()
    false_alerts = flagged.loc[flagged[TARGET] == 0].copy()
    missed = results.loc[
        (results["Predicted_FraudFound_P"] == 0) & (results[TARGET] == 1)
    ].copy()

    st.divider()
    st.subheader("4. Claim-level prediction review")
    st.caption(
        "These are model predictions for the uploaded labelled test records. "
        "A flagged record requires human investigation; it is not a confirmed fraud."
    )

    tab_flagged, tab_correct, tab_false, tab_missed, tab_all = st.tabs(
        [
            f"All flagged ({len(flagged):,})",
            f"Correctly flagged ({len(correctly_flagged):,})",
            f"False alerts ({len(false_alerts):,})",
            f"Missed fraud ({len(missed):,})",
            f"All predictions ({len(results):,})",
        ]
    )

    display_columns = [
        "Evaluation_Row",
        "PolicyNumber",
        TARGET,
        "Predicted_FraudFound_P",
        "Fraud_Probability",
        "Prediction_Status",
        "Actual_Outcome",
    ]
    display_columns = [c for c in display_columns if c in results.columns]

    def show_records(
        frame: pd.DataFrame,
        empty_message: str,
        download_key: str,
    ):
        if frame.empty:
            st.info(empty_message)
            return

        view = frame.loc[:, display_columns].copy()
        view["Fraud_Probability"] = view["Fraud_Probability"].map(
            lambda value: f"{value:.2%}"
        )
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download these records as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{download_key}_claim_records.csv",
            mime="text/csv",
            key=f"download_{download_key}",
        )

    with tab_flagged:
        st.write(
            f"**{len(flagged):,}** records were predicted as fraud and should be "
            "prioritised for review."
        )
        show_records(
            flagged,
            "No records were flagged by this model.",
            "all_flagged",
        )

    with tab_correct:
        st.write(
            f"**{len(correctly_flagged):,}** flagged records have an actual fraud "
            "label in this labelled test file."
        )
        show_records(
            correctly_flagged,
            "No correctly flagged fraud records.",
            "correctly_flagged",
        )

    with tab_false:
        st.write(
            f"**{len(false_alerts):,}** non-fraud records were flagged. These are "
            "false positives in this labelled evaluation sample."
        )
        show_records(
            false_alerts,
            "No false fraud alerts.",
            "false_alerts",
        )

    with tab_missed:
        st.write(
            f"**{len(missed):,}** actual fraud records were predicted as non-fraud."
        )
        show_records(
            missed,
            "No missed fraud records.",
            "missed_fraud",
        )

    with tab_all:
        show_records(
            results,
            "No prediction records are available.",
            "all_predictions",
        )

    st.caption(
        "Evaluation is performed on labelled historical test data. "
        "The actual target column is used only for evaluation and is not passed "
        "to the model as an input feature."
    )

except Exception as error:
    st.error(f"Evaluation could not be completed: {error}")
    st.exception(error)
