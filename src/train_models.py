"""Train the five vehicle-claim fraud classifiers and export evaluation artefacts."""
from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "fraud_oracle.csv"
TARGET = "FraudFound_P"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def build_preprocessor(X):
    numeric_fields = X.select_dtypes(include="number").columns.tolist()
    categorical_fields = X.select_dtypes(exclude="number").columns.tolist()
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("standardise", StandardScaler())
    ])
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, numeric_fields),
        ("categorical", categorical_steps, categorical_fields)
    ], remainder="drop", sparse_threshold=0)

def score_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return predictions, probabilities, {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Place fraud_oracle.csv at {DATA_FILE}")

    claims = pd.read_csv(DATA_FILE)
    if TARGET not in claims.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    if claims[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    if set(claims[TARGET].unique()) != {0, 1}:
        raise ValueError("FraudFound_P must contain binary labels 0 and 1.")

    X = claims.drop(columns=[TARGET, "PolicyNumber"], errors="ignore").copy()
    y = claims[TARGET].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("prepare", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "Decision Tree (Entropy)": Pipeline([
            ("prepare", build_preprocessor(X_train)),
            ("classifier", DecisionTreeClassifier(
                criterion="entropy", max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "KNN": Pipeline([
            ("prepare", build_preprocessor(X_train)),
            ("classifier", KNeighborsClassifier())
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("prepare", build_preprocessor(X_train)),
            ("classifier", GaussianNB())
        ]),
        "SVM": Pipeline([
            ("prepare", build_preprocessor(X_train)),
            ("classifier", SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=RANDOM_STATE
            ))
        ])
    }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    knn_search = GridSearchCV(
        models["KNN"],
        param_grid={"classifier__n_neighbors": [3, 5, 7, 9, 11, 15]},
        scoring="f1", cv=folds, n_jobs=-1
    )
    knn_search.fit(X_train, y_train)
    models["KNN"] = knn_search.best_estimator_

    model_dir, output_dir, data_dir = ROOT / "models", ROOT / "outputs", ROOT / "data"
    for folder in [model_dir, output_dir, data_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    test_records = X_test.copy()
    test_records[TARGET] = y_test.to_numpy()
    test_records.to_csv(data_dir / "test_data.csv", index=False)

    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree (Entropy)": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "SVM": "svm.joblib"
    }
    result_rows = []
    for label, pipeline in models.items():
        if label != "KNN":
            pipeline.fit(X_train, y_train)
        predictions, probabilities, metrics = score_model(pipeline, X_test, y_test)
        result_rows.append({"Model": label, **metrics})
        joblib.dump(pipeline, model_dir / filenames[label])
        print(f"\n{label}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("Fraud predictions:", int((predictions == 1).sum()))

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    metadata = {
        "dataset_rows": int(len(claims)),
        "dataset_columns_including_target": int(claims.shape[1]),
        "predictor_count": int(X.shape[1]),
        "target": TARGET,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "knn_selected_k": int(knn_search.best_params_["classifier__n_neighbors"]),
        "model_selection_rule": "Highest F1; tie-break by MCC, then ROC-AUC"
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved five model pipelines, test_data.csv and evaluation outputs.")

if __name__ == "__main__":
    main()
