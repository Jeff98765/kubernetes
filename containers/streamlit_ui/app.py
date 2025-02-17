import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Get the absolute path of the current script
base_path = os.path.dirname(os.path.abspath(__file__))

# Ensure Python can find the 'containers' directory
sys.path.append(os.path.abspath(os.path.join(base_path, "..")))

# Import necessary functions
from dataprep.dataprep_c1 import preprocess_dataset
from model.model_c2 import split_data, train_and_evaluate_models, save_model
from prediction.prediction_c3 import make_predictions

# Initialize session state variables if not already set
if "train_data" not in st.session_state:
    st.session_state.train_data = None
if "processed_train_data" not in st.session_state:
    st.session_state.processed_train_data = None
if "test_data" not in st.session_state:
    st.session_state.test_data = None
if "processed_test_data" not in st.session_state:
    st.session_state.processed_test_data = None
if "model" not in st.session_state:
    st.session_state.model = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

st.title("AI Model Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Preprocessing", "Modeling", "Prediction"])

# ------------------ DATA PREPROCESSING ------------------
if page == "Data Preprocessing":
    st.header("Upload Train and Test Data for Preprocessing")
    
    uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
    uploaded_test = st.file_uploader("Upload Test CSV", type=["csv"], key="test")

    if uploaded_train is not None:
        st.session_state.train_data = pd.read_csv(uploaded_train)
        st.session_state.processed_train_data = preprocess_dataset(st.session_state.train_data)
        st.success("Training data preprocessed successfully!")
        st.write(st.session_state.processed_train_data)

    if uploaded_test is not None:
        st.session_state.test_data = pd.read_csv(uploaded_test)
        st.session_state.processed_test_data = preprocess_dataset(st.session_state.test_data)
        st.success("Test data preprocessed successfully!")
        st.write(st.session_state.processed_test_data)

# ------------------ MODELING (TRAINING & EVALUATION) ------------------
elif page == "Modeling":
    st.header("Train & Evaluate Model")

    if st.session_state.processed_train_data is not None:
        st.write("Splitting data, training, and evaluating the model...")

        # Split training data into train-test
        X_train, X_test, y_train, y_test = split_data(st.session_state.processed_train_data)

        # Train and evaluate the model
        best_model_name, best_model, results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        
        # Save the model
        save_model(best_model, best_model_name)

        # Store in session state
        st.session_state.model = best_model
        st.session_state.best_model_name = best_model_name

        st.success(f"Model trained successfully: {best_model_name}")
        st.write("Model Evaluation Metrics:", results)
    else:
        st.warning("Please preprocess the train data first in the Data Preprocessing section.")

# ------------------ PREDICTION ON TEST DATA ------------------
elif page == "Prediction":
    st.header("Generate Predictions on Test Dataset")

    if st.session_state.model is None:
        st.warning("No trained model found. Train the model first.")
    elif st.session_state.processed_test_data is not None:
        st.write("Running model on actual test dataset...")

        X_test = st.session_state.processed_test_data.drop(columns=["Survived"], errors='ignore')
        
        # Use the stored model name to locate the saved model
        model_path = f"saved_model/{st.session_state.best_model_name.lower().replace(' ', '_')}_model.pkl"
        output_filename = "final_predictions.csv"

        make_predictions(X_test, model_path, output_filename)

        st.success("Predictions saved successfully!")
        st.write(f"Predictions saved to: {output_filename}")
    else:
        st.warning("Please preprocess the test data first in the Data Preprocessing section.")
