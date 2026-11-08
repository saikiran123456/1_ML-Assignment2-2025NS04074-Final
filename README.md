# Vehicle Insurance Claim Fraud Detection

## Problem statement
This project evaluates whether historical vehicle insurance claim records can be classified as potentially fraudulent (`FraudFound_P = 1`) or non-fraudulent (`FraudFound_P = 0`). The output is intended only as a screening signal to support human review; it is not proof of fraud and must not be used for automatic claim rejection.

## Dataset description
- Source: [Kaggle – Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- Records: 15,420
- Columns including target: 33
- Predictors used: 31 (the target and `PolicyNumber` identifier are excluded)
- Target: `FraudFound_P` (0 = non-fraud, 1 = fraud)
- Class counts: non-fraud = 14,497; fraud = 923
- Split: stratified 80:20; random state = 42

This is a historical vehicle-insurance dataset. It is not proprietary data from ICICI Lombard and is not described as an Indian insurer dataset.

## GitHub repository
Add the public repository URL here after creating and verifying the repository.

## Models and evaluation metrics
All five classifiers use the same training/test split. Metrics are calculated for the fraud class as the positive class. AUC is binary ROC-AUC.

                  Model  Accuracy    AUC  Precision  Recall     F1    MCC
    Logistic Regression    0.6440 0.8063     0.1322  0.8865 0.2300 0.2494
Decision Tree (Entropy)    0.6540 0.7979     0.1355  0.8865 0.2351 0.2557
                    KNN    0.9248 0.5948     0.1385  0.0486 0.0720 0.0485
   Gaussian Naive Bayes    0.0973 0.5387     0.0608  0.9730 0.1145 0.0173
                    SVM    0.7866 0.8290     0.1738  0.6811 0.2769 0.2657

## Model observations
- **Logistic Regression:** Accuracy 0.6440, AUC 0.8063, Precision 0.1322, Recall 0.8865, F1 0.2300, MCC 0.2494. Its results should be interpreted alongside the fraud-class precision and recall.
- **Decision Tree (Entropy):** Accuracy 0.6540, AUC 0.7979, Precision 0.1355, Recall 0.8865, F1 0.2351, MCC 0.2557. Its results should be interpreted alongside the fraud-class precision and recall.
- **KNN:** Accuracy 0.9248, AUC 0.5948, Precision 0.1385, Recall 0.0486, F1 0.0720, MCC 0.0485. Its results should be interpreted alongside the fraud-class precision and recall.
- **Gaussian Naive Bayes:** Accuracy 0.0973, AUC 0.5387, Precision 0.0608, Recall 0.9730, F1 0.1145, MCC 0.0173. Its results should be interpreted alongside the fraud-class precision and recall.
- **SVM:** Accuracy 0.7866, AUC 0.8290, Precision 0.1738, Recall 0.6811, F1 0.2769, MCC 0.2657. This model ranked first under the F1/MCC/AUC selection rule.

## Overall model selection
The highest-ranked model under the pre-defined rule (F1, followed by MCC and AUC) is **SVM** (F1 = 0.2769, MCC = 0.2657, AUC = 0.8290). This ranking is specific to this dataset and held-out split. The model should be treated as a candidate for further validation, not a production-ready fraud decision system.

## Repository contents
- `notebooks/Vehicle_Insurance_Fraud_Detection.ipynb`
- `src/train_models.py`
- `app.py`
- `requirements.txt`
- `data/test_data.csv`
- `model/` (five fitted preprocessing/classifier pipelines; required repository folder)
- `models/` (duplicate artefacts retained for compatibility with the Streamlit application)
- `outputs/model_comparison.csv`
