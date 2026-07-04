import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="SaaS Churn Intelligence Engine", 
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_production_pipeline():
    return joblib.load("models/saas_churn_pipeline_v2.joblib")

pipeline = load_production_pipeline()

# Modern Header Card
st.markdown("""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border-left: 5px solid #FF6B35; margin-bottom: 25px;">
        <h1 style="margin: 0; color: #F8FAFC;">📊 SaaS Customer Churn & Retention Intelligence System</h1>
        <p style="margin: 5px 0 0 0; color: #94A3B8;">Optimized Gradient Boosting engine to evaluate customer telemetry, predict churn probability, and isolate risk factors.</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# NEW: VIDEOGRAPHIC PIPELINE ARCHITECTURE REPRESENTATION
# ==========================================
with st.expander("🔍 System Architecture: How the Production Model Works", expanded=True):
    st.markdown("### ⚙️ Self-Contained End-to-End Inference Pipeline")
    st.caption("When you submit data below, it moves sequentially through this fully containerized blueprint:")
    
    # Render responsive architectural nodes using Streamlit metric layout columns
    flow_col1, arrow1, flow_col2, arrow2, flow_col3, arrow3, flow_col4 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
    
    with flow_col1:
        st.metric(label="📥 Input Layer", value="Raw Telemetry Data", delta="Mixed Data Types")
        st.markdown("<small style='color: #94A3B8;'>Accepts numerical hours/spend alongside categorical text strings seamlessly.</small>", unsafe_allow_html=True)
        
    with arrow1:
        st.markdown("<h1 style='text-align: center; color: #FF6B35; padding-top: 10px;'>➔</h1>", unsafe_allow_html=True)
        
    with flow_col2:
        st.metric(label="🔧 ColumnTransformer", value="Parallel Preprocessing", delta="Zero Leakage")
        st.markdown("<small style='color: #94A3B8;'>Splits numerical parameters into scaling algorithms while text inputs route into isolated OneHot encoders automatically.</small>", unsafe_allow_html=True)
        
    with arrow2:
        st.markdown("<h1 style='text-align: center; color: #FF6B35; padding-top: 10px;'>➔</h1>", unsafe_allow_html=True)
        
    with flow_col3:
        st.metric(label="🧠 Model Brain", value="HistGradientBoosting", delta="Optuna Optimized")
        st.markdown("<small style='color: #94A3B8;'>Evaluates complex telemetry via ensembles of regularized decision trees utilizing balanced minority class weights.</small>", unsafe_allow_html=True)
        
    with arrow3:
        st.markdown("<h1 style='text-align: center; color: #FF6B35; padding-top: 10px;'>➔</h1>", unsafe_allow_html=True)
        
    with flow_col4:
        st.metric(label="🔮 Output Layer", value="Probability Score", delta="Actionable Risk")
        st.markdown("<small style='color: #94A3B8;'>Generates definitive risk tags alongside custom retention logic strategies based on profile confidence values.</small>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initiate operational tabs (Keep the remaining Tab 1 and Tab 2 logic exactly as it was)
tab1, tab2 = st.tabs(["📁 Fleet Batch CSV Analysis", "👤 Single Customer Lookup"])