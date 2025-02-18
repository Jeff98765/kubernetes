import streamlit as st
import pandas as pd
import requests
import os

# ------------------ 🌟 Initialize Session State ------------------
for var in ["train_data", "processed_train_data", "test_data", "processed_test_data", "model", "best_model_name", "predictions_file"]:
    if var not in st.session_state:
        st.session_state[var] = None

# ------------------ 📂 File Upload Section ------------------
st.title("💜 AI Pipeline Dashboard")
st.markdown("<h4 style='text-align: center; color: gray;'>An Automated End-to-End Machine Learning Pipeline</h4>", unsafe_allow_html=True)

uploaded_train = st.file_uploader("📂 Upload Training CSV", type=["csv"], key="train")
uploaded_test = st.file_uploader("📂 Upload Test CSV", type=["csv"], key="test")

if uploaded_train and uploaded_test:
    st.subheader("🚀 *Starting Automated Pipeline...*")
    progress_bar = st.progress(0)
    
    # Ensure directories exist
    os.makedirs("data/01_raw", exist_ok=True)
    os.makedirs("data/02_processed", exist_ok=True)
    os.makedirs("data/03_predicted", exist_ok=True)
    
    # Save uploaded files
    train_path = "data/01_raw/train.csv"
    test_path = "data/01_raw/predict.csv"
    pd.read_csv(uploaded_train).to_csv(train_path, index=False)
    pd.read_csv(uploaded_test).to_csv(test_path, index=False)
    
    progress_bar.progress(20)
    
    # ------------------ 🔄 Data Preprocessing ------------------
    st.write("<h4 style='color: #8A2BE2;'>🔄 Preprocessing Data...</h4>", unsafe_allow_html=True)
    preprocess_url = "http://127.0.0.1:5000/preprocess"
    try:
        response = requests.get(preprocess_url)
        response.raise_for_status()
        st.success("✅ *Data Preprocessing Completed!*")
        progress_bar.progress(50)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ *Preprocessing failed:* {e}")

    # ------------------ 🔄 Model Training ------------------
    st.write("<h4 style='color: #4B0082;'>🔄 Training and Evaluating Model...</h4>", unsafe_allow_html=True)
    model_url = "http://127.0.0.1:5001/model"
    try:
        with open("data/02_processed/train_processed.csv", 'rb') as file1, open("data/02_processed/predict_processed.csv", 'rb') as file2:
            response = requests.post(model_url, files={'file1': file1, 'file2': file2})
        response.raise_for_status()
        st.success("✅ *Model Training Completed!*")
        model_response = response.json()
        st.json(model_response)
        progress_bar.progress(80)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ *Model Training failed:* {e}")
    
    # ------------------ 🔄 Running Predictions ------------------
    st.write("<h4 style='color: #6A0DAD;'>🔄 Running Predictions...</h4>", unsafe_allow_html=True)
    predict_url = "http://127.0.0.1:5002/predict"
    try:
        with open("data/02_processed/predict_processed.csv", 'rb') as file:
            response = requests.post(predict_url, files={'file': file})
        response.raise_for_status()
        pred_output = response.json().get("output_file", "")
        if pred_output:
            st.session_state.predictions_file = pred_output
            st.success("✅ *Predictions Generated!*")
            progress_bar.progress(100)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ *Prediction failed:* {e}")
    
    # ------------------ 📌 Display Predictions ------------------
    if st.session_state.predictions_file:
        st.subheader("📌 Final Predictions")
        try:
            predictions_df = pd.read_csv(st.session_state.predictions_file)
            st.dataframe(predictions_df)
            st.download_button("📥 Download Predictions CSV", data=predictions_df.to_csv(index=False), file_name="final_predictions.csv", mime="text/csv")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Error loading predictions: {e}")
