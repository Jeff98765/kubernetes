import streamlit as st
import pandas as pd
import joblib
import os
import sys
import subprocess
from sklearn.metrics import accuracy_score

# Ensure Python recognizes the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define paths to container scripts
DATAPREP_SCRIPT = "../dataprep/dataprep_c1.py"
MODEL_SCRIPT = "../model/model_c2.py"
PREDICTION_SCRIPT = "../prediction/prediction_c3.py"

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

st.title("AI Model Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Preprocessing", "Model Training", "Model Evaluation"])

if page == "Data Preprocessing":
    st.header("Upload Train Data for Preprocessing")
    uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
    if uploaded_train is not None:
        train_path = "train_data.csv"
        with open(train_path, "wb") as f:
            f.write(uploaded_train.getbuffer())
        
        # Run the data preprocessing container
        subprocess.run(["python", DATAPREP_SCRIPT, train_path], check=True)
        st.success("Data preprocessing complete!")

elif page == "Model Training":
    st.header("Train Model")
    if st.session_state.processed_train_data is not None:
        st.write("Training model...")
        subprocess.run(["python", MODEL_SCRIPT, "processed_train_data.csv"], check=True)
        st.success("Model trained successfully!")
    else:
        st.warning("Please preprocess data first in the Data Preprocessing section.")
    
elif page == "Model Evaluation":
    st.header("Evaluate Model Accuracy")
    uploaded_test = st.file_uploader("Upload Test Data", type=["csv"], key="test")
    model_path = "../model/trained_model.pkl"
    
    if uploaded_test is not None:
        test_path = "test_data.csv"
        with open(test_path, "wb") as f:
            f.write(uploaded_test.getbuffer())
        
        # Run the prediction container
        subprocess.run(["python", PREDICTION_SCRIPT, test_path, model_path], check=True)
        st.success("Model evaluation complete!")
