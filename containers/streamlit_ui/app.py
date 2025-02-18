import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Get the absolute path of the current script
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(base_path, "..", "data", "03_predicted"))

# Ensure Python can find the 'containers' directory
sys.path.append(os.path.abspath(os.path.join(base_path, "..")))

# Import necessary functions
from dataprep.dataprep_c1 import preprocess_dataset
from model.model_c2 import split_data, train_and_evaluate_models, save_model
from prediction.prediction_c3 import make_predictions

# Initialize session state variables
for var in ["train_data", "processed_train_data", "test_data", "processed_test_data", "model", "best_model_name", "predictions_file"]:
    if var not in st.session_state:
        st.session_state[var] = None

# Ensure prediction directory exists
os.makedirs(data_path, exist_ok=True)

# ------------------ 🎨 Custom Styles ------------------
st.markdown(
    """
    <style>
        /* General Background */
        .stApp {
            background-color: #E6E6FA !important;  /* Soft Lavender */
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #D8BFD8 !important;  /* Light Mauve */
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #4B0082 !important;  /* Deep Plum */
        }

        /* Buttons */
        .stButton>button {
            background-color: #BA55D3 !important; /* Dusty Purple */
            color: white !important;
            border-radius: 10px;
        }

        /* File Upload Background */
        div[data-testid="stFileUploader"] {
            background-color: #D8BFD8 !important;
            padding: 10px;
            border-radius: 10px;
        }

        /* Download Button */
        .stDownloadButton>button {
            background-color: #BA55D3 !important; /* Orchid */
            color: white !important;
            border-radius: 8px;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ 🏠 Sidebar Navigation ------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png", width=150)
st.sidebar.title("🔍 AI Pipeline Navigation")
st.sidebar.markdown("<h3 style='color: #4B0082;'>Automated Machine Learning Workflow</h3>", unsafe_allow_html=True)
st.sidebar.divider()

st.sidebar.info(
    """
    🚀 **Steps in Pipeline**
    - Upload Train & Test Data
    - Auto Preprocessing 🛠
    - Train Best Model 📊
    - Generate Predictions ✅
    """
)

# ------------------ 🔥 Main Web App ------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4B0082;'>💜 AI Pipeline Dashboard</h1>
    <h4 style='text-align: center; color: gray;'>An Automated End-to-End Machine Learning Pipeline</h4>
    """, unsafe_allow_html=True
)

# ------------------ 📂 File Uploads ------------------
st.markdown("<h2 style='color: #BA55D3;'>📤 Upload Train & Test Data to Start Pipeline</h2>", unsafe_allow_html=True)

uploaded_train = st.file_uploader("📂 Upload Training CSV", type=["csv"], key="train")
uploaded_test = st.file_uploader("📂 Upload Test CSV", type=["csv"], key="test")

if uploaded_train and uploaded_test:
    st.subheader("🚀 **Starting Automated Pipeline...**")
    
    # 1️. **Preprocessing**
    st.write("<h4 style='color: #8A2BE2;'>🔄 Preprocessing Data...</h4>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    st.session_state.train_data = pd.read_csv(uploaded_train)
    st.session_state.test_data = pd.read_csv(uploaded_test)
    
    st.session_state.processed_train_data = preprocess_dataset(st.session_state.train_data)
    st.session_state.processed_test_data = preprocess_dataset(st.session_state.test_data)

    st.success("✅ **Data Preprocessing Completed!**")
    st.dataframe(st.session_state.processed_train_data.head())

    progress_bar.progress(30)

    # 2️. **Model Training & Evaluation**
    st.write("<h4 style='color: #4B0082;'>🔄 Training and Evaluating Model...</h4>", unsafe_allow_html=True)
    
    X_train, X_test, y_train, y_test = split_data(st.session_state.processed_train_data)
    
    best_model_name, best_model, results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    save_model(best_model, best_model_name)

    st.session_state.model = best_model
    st.session_state.best_model_name = best_model_name

    st.success(f"✅ **Model Training Completed!** Best Model: `{best_model_name}`")
    st.subheader("📊 Evaluation Metrics")
    st.json(results)

    progress_bar.progress(70)

    # 3️. **Run Predictions on Test Data**
    st.write("<h4 style='color: #6A0DAD;'>🔄 Running Model on Test Dataset...</h4>", unsafe_allow_html=True)

    X_test_final = st.session_state.processed_test_data.drop(columns=["Survived"], errors='ignore')

    model_path = f"saved_model/{st.session_state.best_model_name.lower().replace(' ', '_')}_model.pkl"
    predictions_file = os.path.join(data_path, "final_predictions.csv")

    make_predictions(X_test_final, model_path, predictions_file)

    st.session_state.predictions_file = predictions_file
    st.success(f"✅ **Predictions Generated & Saved in `{predictions_file}`!**")

    progress_bar.progress(100)

    # 4️. **Display Predictions & Download Option**
    st.header("📌 Final Predictions")

    if os.path.exists(predictions_file):
        predictions_df = pd.read_csv(predictions_file)
        st.dataframe(predictions_df)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=predictions_df.to_csv(index=False),
            file_name="final_predictions.csv",
            mime="text/csv"
        )
        st.balloons()  # 🎉 Celebration Animation
    else:
        st.error("❌ **Predictions file not found.** Check the model output.")