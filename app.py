from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
TARGET = "FraudFound_P"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree (Entropy)": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

st.set_page_config(page_title="ClaimLens | Fraud Analytics", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 14px; background: linear-gradient(110deg,#10243A,#1D4967); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#D8E6F0; margin:.35rem 0 0 0;}
div[data-testid="stMetric"] {background:#F4F7FA; border:1px solid #DCE5ED; padding:12px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>ClaimLens | Vehicle Claim Fraud Analytics</h1>
<p>Comparative evaluation of machine-learning classifiers for claim screening.</p>
</div>
""", unsafe_allow_html=True)
st.warning("A model flag is a screening indicator only. It is not proof of fraud and must not be used as an automatic reason to reject a claim.")

with st.sidebar:
    st.header("Evaluation setup")
    model_label = st.selectbox("Classification model", list(MODEL_FILES))
    st.caption("Evaluation uses a labelled CSV so predicted classes can be compared with known outcomes.")
    st.markdown("---")
    st.caption("Target: FraudFound_P · 0 = Non-fraud · 1 = Fraud")

model_path = ROOT / "models" / MODEL_FILES[model_label]

@st.cache_resource
def load_pipeline(path_string):
    return joblib.load(path_string)

st.subheader("Upload evaluation data")
uploaded_file = st.file_uploader(
    "Choose a labelled CSV file",
    type=["csv"],
    help="Use the generated data/test_data.csv. It must include all model input columns and FraudFound_P."
)
if uploaded_file is None:
    st.info("Upload the labelled test dataset to view the selected model's evaluation.")
    sample_path = ROOT / "data" / "test_data.csv"
    if sample_path.exists():
        st.caption("Sample evaluation file is available at data/test_data.csv.")
    st.stop()

if not model_path.exists():
    st.error(f"Trained model not found: {model_path.name}. Run the notebook or `python src/train_models.py` first.")
    st.stop()

try:
    evaluation_data = pd.read_csv(uploaded_file)
    if evaluation_data.empty:
        st.error("The uploaded CSV contains no records.")
        st.stop()
    if TARGET not in evaluation_data.columns:
        st.error(f"The uploaded CSV must contain the actual target column '{TARGET}'.")
        st.stop()

    actual = evaluation_data[TARGET].astype(int)
    if not set(actual.unique()).issubset({0, 1}):
        st.error("FraudFound_P must contain only 0 and 1.")
        st.stop()

    pipeline = load_pipeline(str(model_path))
    expected_features = list(pipeline.feature_names_in_)
    provided_features = [column for column in evaluation_data.columns if column != TARGET]
    missing_features = sorted(set(expected_features) - set(provided_features))
    if missing_features:
        st.error("Required feature columns are missing: " + ", ".join(missing_features))
        st.stop()

    feature_data = evaluation_data[expected_features]
    predicted = pipeline.predict(feature_data)
    probability = pipeline.predict_proba(feature_data)[:, 1]

    st.caption(f"Model: {model_label}  |  Evaluation records: {len(evaluation_data):,}")
    st.subheader("Evaluation metrics")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Accuracy", f"{accuracy_score(actual, predicted):.4f}")
    metric_columns[1].metric("ROC-AUC", f"{roc_auc_score(actual, probability):.4f}")
    metric_columns[2].metric("MCC", f"{matthews_corrcoef(actual, predicted):.4f}")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Precision", f"{precision_score(actual, predicted, zero_division=0):.4f}")
    metric_columns[1].metric("Recall", f"{recall_score(actual, predicted, zero_division=0):.4f}")
    metric_columns[2].metric("F1-score", f"{f1_score(actual, predicted, zero_division=0):.4f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual non-fraud", "Actual fraud"],
                columns=["Predicted non-fraud", "Predicted fraud"]
            ),
            use_container_width=True
        )
    with right:
        st.subheader("Classification report")
        report = classification_report(
            actual, predicted,
            labels=[0, 1],
            target_names=["Non-fraud", "Fraud"],
            output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    with st.expander("Preview uploaded records"):
        st.dataframe(evaluation_data.head(20), use_container_width=True)
except Exception as error:
    st.error(f"Unable to evaluate the uploaded file: {error}")
