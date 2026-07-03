import streamlit as st
import joblib
import numpy as np

# 1. Setup the page
st.set_page_config(page_title="ML Tuning Project", page_icon="🚀")
st.title("Random Forest Predictor 🌲")
st.markdown("This app uses a hyperparameter-tuned Random Forest model to classify data based on 4 inputs.")

# 2. Load the model 
# @st.cache_resource ensures the model only loads once, keeping the app fast
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_v1.joblib")

model = load_model()

# 3. Create the user interface
st.header("Input Features")

# Put inputs side-by-side using columns
col1, col2 = st.columns(2)
with col1:
    f1 = st.number_input("Feature 1 (e.g., 0.5)", value=0.0)
    f2 = st.number_input("Feature 2 (e.g., -1.2)", value=0.0)
with col2:
    f3 = st.number_input("Feature 3 (e.g., 3.4)", value=0.0)
    f4 = st.number_input("Feature 4 (e.g., 0.1)", value=0.0)

# 4. The Prediction Engine
if st.button("Make Prediction", type="primary"):
    # Format the UI inputs into the 2D array the model expects
    input_data = np.array([[f1, f2, f3, f4]])
    
    # Ask the model
    prediction = model.predict(input_data)
    
    # Display the result to the user
    st.success(f"The model predicts: **Class {prediction[0]}**")
    st.balloons() # Add a little celebration!