# --- File: app.py (Corrected Version) ---

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost 
import os     

# --- Caching the Model and Feature Names ---
@st.cache_resource
def load_model_and_features():
    """Load the saved model and feature names from the 'model' directory."""
    # --- THIS IS THE KEY CORRECTION ---
    # We define the correct path to the 'model' folder.
    MODEL_DIR = 'model'
    MODEL_PATH = os.path.join(MODEL_DIR, 'optimized_xgb_model.pkl')
    FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, feature_names
    except FileNotFoundError:
        # This error message is now more helpful
        st.error(f"Error: Model or feature files not found. Please ensure the following files exist:\n"
                 f"1. `{MODEL_PATH}`\n"
                 f"2. `{FEATURES_PATH}`\n"
                 "Run the Jupyter Notebook to generate them.")
        return None, None

model, feature_names = load_model_and_features()

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered"
)


# --- Main Application ---
# The rest of the code is the same, but it will only run if the model loads successfully.
if model and feature_names:
    st.title("📈 Advanced Sales Predictor")
    st.write(
        "Welcome! This app predicts sales revenue based on advertising budgets for TV, Radio, and Newspaper. "
        "Use the sliders in the sidebar to set your budget."
    )

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Advertising Budgets ($)")
    tv_spend = st.sidebar.slider("TV Advertising Spend", min_value=0, max_value=300, value=150, step=5)
    radio_spend = st.sidebar.slider("Radio Advertising Spend", min_value=0, max_value=50, value=25, step=1)
    newspaper_spend = st.sidebar.slider("Newspaper Advertising Spend", min_value=0, max_value=120, value=10, step=1)

    # --- Feature Engineering ---
    total_spend = tv_spend + radio_spend + newspaper_spend
    input_data = {
        'TV': tv_spend,
        'Radio': radio_spend,
        'TV_Radio_Interaction': tv_spend * radio_spend,
        'TV_sq': tv_spend**2,
        'Radio_sq': radio_spend**2,
        'TV_Share': np.divide(tv_spend, total_spend, where=total_spend!=0),
        'Radio_Share': np.divide(radio_spend, total_spend, where=total_spend!=0)
    }
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # --- Prediction and Display ---
    if st.sidebar.button("Predict Sales", type="primary"):
        prediction = model.predict(input_df)[0]
        st.subheader("🔮 Predicted Sales Revenue")
        st.markdown(
            f"""
            <div style="background-color: #D3F3E3; padding: 20px; border-radius: 10px;">
                <h2 style="color: #0E6F50; text-align: center;">${prediction:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander("Show Model Inputs and Feature Importances"):
            st.write("**Engineered Features Sent to Model:**")
            st.dataframe(input_df.T.rename(columns={0: 'Value'}))
            st.write("\n**Model's Feature Importance:**")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.bar_chart(importance_df.set_index('Feature'))