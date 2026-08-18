# app.py
# Streamlit Application for Credit Default Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    matthews_corrcoef
from sklearn.metrics import classification_report

# Import custom preprocessing functions
from src.utils.data_processor import engineer_risk_features

# Page configuration
st.set_page_config(page_title="Credit Default Analyzer", layout="wide")
st.title("💳 Credit Default Risk Analyzer")
st.markdown("Upload your test data and evaluate 5 classification models.")


# Load saved models, scaler, and feature columns
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load('models/logistic_risk_v1.joblib'),
        "Decision Tree": joblib.load('models/decision_tree_v1.joblib'),
        "KNN": joblib.load('models/knn_v1.joblib'),
        "Naive Bayes": joblib.load('models/naive_bayes_v1.joblib'),
        "Random Forest": joblib.load('models/random_forest_v1.joblib')
    }
    scaler = joblib.load('models/scaler.joblib')
    feature_cols = joblib.load('models/feature_columns.joblib')
    return models, scaler, feature_cols


models, scaler, feature_cols = load_models()

# Sidebar: File upload and model selection
with st.sidebar:
    st.header("📁 Upload Test Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    st.header("🤖 Select Model")
    selected_model_name = st.selectbox("Choose a model", list(models.keys()))

    st.markdown("---")
    st.markdown("**Dataset:** UCI Credit Card Default")
    st.markdown("**Target:** default_flag (1 = default, 0 = no default)")
    st.markdown("**Note:** Upload test data with the target column 'default_flag'.")

# Main content
if uploaded_file is not None:
    # Load the uploaded file
    test_df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Test Data Preview")
    st.dataframe(test_df.head())

    # Check if target column exists
    if 'default_flag' not in test_df.columns:
        st.error("❌ The uploaded file must contain a 'default_flag' column (0 = no default, 1 = default).")
    else:
        # Separate features and target
        y_true = test_df['default_flag'].values
        test_df_features = test_df.drop(columns=['default_flag', 'client_unique_id'], errors='ignore')

        # Apply feature engineering (same as training)
        try:
            test_df_engineered = engineer_risk_features(test_df_features)
        except Exception as e:
            st.error(f"❌ Error in feature engineering: {e}")
            st.stop()

        # Ensure all required features are present
        missing_cols = set(feature_cols) - set(test_df_engineered.columns)
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        # Reorder columns to match training order
        test_df_engineered = test_df_engineered[feature_cols]

        # Get the numerical columns that were scaled during training
        scaled_cols = scaler.feature_names_in_  # list of column names the scaler was fitted on

        # Scale only those numerical columns
        test_df_engineered[scaled_cols] = scaler.transform(test_df_engineered[scaled_cols])

        # Get the selected model
        model = models[selected_model_name]

        # Predict using the full feature set (now with scaled numerical columns)
        y_pred = model.forecast_default_class(test_df_engineered)
        y_proba = model.forecast_default_proba(test_df_engineered)

        # Compute metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_proba),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }

        # Display metrics
        st.subheader(f"📈 Evaluation Metrics for {selected_model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col1.metric("AUC", f"{metrics['AUC']:.4f}")
        col2.metric("Precision", f"{metrics['Precision']:.4f}")
        col2.metric("Recall", f"{metrics['Recall']:.4f}")
        col3.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        col3.metric("MCC", f"{metrics['MCC']:.4f}")

        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['No Default', 'Default'])
        ax.set_yticklabels(['No Default', 'Default'])
        st.pyplot(fig)

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_true, y_pred, target_names=['No Default', 'Default'])
        st.text(report)

else:
    st.info("👈 Please upload a CSV file to begin analysis.")

# Footer
st.markdown("---")
st.caption("Built for BITS Pilani ML Assignment 2 | Credit Default Dataset (UCI)")
