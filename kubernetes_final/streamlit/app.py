import streamlit as st
import pandas as pd
import requests
import os
import time

# ------------------ 🌟 Initialize Session State ------------------
for var in ["train_data", "processed_train_data", "test_data", "processed_test_data", "model", "best_model_name", "predictions_file"]:
    if var not in st.session_state:
        st.session_state[var] = None

# ------------------ 🎨 UI Setup ------------------
st.set_page_config(page_title="AI Pipeline Dashboard", page_icon="🌸", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #ffe6f2;
            color: #ff1493;
        }
        .stApp {
            background-color: #ffe6f2;
        }
        .css-1d391kg p, h4, h3, h2, h1 {
            color: #ff1493 !important;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #ff66b2;
            color: white;
            border-radius: 10px;
            font-size: 16px;
        }
        .stProgress>div>div>div {
            background-color: #ff66b2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌸 AI Pipeline Dashboard")
st.markdown("<h4 style='text-align: center; color: #ff1493;'>Your Personalized Machine Learning Workflow</h4>", unsafe_allow_html=True)

st.divider()

# ------------------ 📂 File Upload Section ------------------
st.subheader("📤 Upload Your Dataset")
col1, col2 = st.columns(2)
with col1:
    uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
with col2:
    uploaded_test = st.file_uploader("Upload Test CSV", type=["csv"], key="test")

if uploaded_train and uploaded_test:
    st.subheader("✨ Running AI Pipeline...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Ensure directories exist
    os.makedirs("data/01_raw", exist_ok=True)
    os.makedirs("data/02_processed", exist_ok=True)
    os.makedirs("data/03_predicted", exist_ok=True)
    
    # Save uploaded files
    train_path = "data/01_raw/train.csv"
    test_path = "data/01_raw/predict.csv"
    pd.read_csv(uploaded_train).to_csv(train_path, index=False)
    pd.read_csv(uploaded_test).to_csv(test_path, index=False)
    
    progress_bar.progress(10)
    status_text.write("📂 Files uploaded successfully!")
    time.sleep(1)

    # ------------------ 🔄 Data Preprocessing ------------------
    with st.spinner("🌸 Preprocessing Data..."):
        preprocess_url = "http://dataprep-service:80/preprocess"
        try:
            response = requests.get(preprocess_url)
            response.raise_for_status()
            st.success("🎉 Data Preprocessing Completed!")
            progress_bar.progress(40)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Preprocessing failed: {e}")
            st.stop()
    time.sleep(1)
    
    # ------------------ 🔄 Model Training ------------------
    with st.spinner("🤖 Training AI Model..."):
        model_url = "http://model-service:80/model"
        try:
            with open("data/02_processed/train_processed.csv", 'rb') as file1, open("data/02_processed/predict_processed.csv", 'rb') as file2:
                response = requests.post(model_url, files={'file1': file1, 'file2': file2})
            response.raise_for_status()
            st.success("🎉 Model Training Completed!")
            model_response = response.json()
            st.json(model_response)
            progress_bar.progress(70)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Model Training failed: {e}")
            st.stop()
    time.sleep(1)
    
    # ------------------ 🔄 Running Predictions ------------------
    with st.spinner("🔮 Running AI Predictions..."):
        predict_url = "http://prediction-service:80/predict"
        try:
            with open("data/02_processed/predict_processed.csv", 'rb') as file:
                response = requests.post(predict_url, files={'file': file})
            response.raise_for_status()
            pred_output = response.json().get("output_file", "")
            if pred_output:
                st.session_state.predictions_file = pred_output
                st.success("🎉 Predictions Generated!")
                progress_bar.progress(100)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()
    time.sleep(1)
    
    # ------------------ 📌 Display Predictions ------------------
    if st.session_state.predictions_file:
        st.subheader("📌 Final Predictions")
        try:
            predictions_df = pd.read_csv(st.session_state.predictions_file)
            st.dataframe(predictions_df)
            st.download_button("📥 Download Predictions CSV", data=predictions_df.to_csv(index=False), file_name="final_predictions.csv", mime="text/csv")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Error loading predictions: {e}")