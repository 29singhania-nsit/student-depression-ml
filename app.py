import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


st.title("Student Depression Classification")

st.write(
    "Upload the test dataset and select a machine learning model."
)


model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl"
}


selected_model = st.selectbox(
    "Select Classification Model",
    list(model_files.keys())
)


uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(test_data.head())

    X_test = test_data.drop("Depression", axis=1)

    y_test = test_data["Depression"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )

    model = joblib.load(
        model_files[selected_model]
    )

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )


    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No",
            "Actual Yes"
        ],
        columns=[
            "Predicted No",
            "Predicted Yes"
        ]
    )

    st.subheader("Confusion Matrix")

    st.dataframe(cm_df)