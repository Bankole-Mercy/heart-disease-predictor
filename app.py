import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train and save the model first.")
        st.stop()

model, scaler = load_model()

# Title
st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Machine Learning (Logistic Regression)")
st.markdown("---")

# Sidebar - Information
with st.sidebar:
    st.header("📊 About This App")
    st.info("""
    This app uses a **Logistic Regression** model trained on cardiovascular health data.
    
    **Model Performance:**
    - Accuracy: 88%
    - Precision: 88%
    - Recall: 86%
    - F1-Score: 87%
    
    **Created by:** Mercy  
    **Institution:** Northeastern University  
    **Program:** MS in Information Systems
    """)
    
    st.header("⚕️ Disclaimer")
    st.warning("""
    This tool is for educational purposes only. 
    It does NOT replace professional medical advice.
    Please consult a healthcare provider for proper diagnosis.
    """)

# Main input form
st.markdown('<h2 class="sub-header">Patient Information Form</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", 
                               options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120, step=1)

with col2:
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    rest_ecg = st.selectbox("Resting ECG", 
                            options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150, step=1)

with col3:
    exercise_angina = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", options=["Upsloping", "Flat", "Downsloping"])
    num_vessels = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thallium = st.selectbox("Thallium Stress Test", options=["Normal", "Fixed Defect", "Reversible Defect"])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
    # Encode categorical variables
    sex_encoded = 1 if sex == "Male" else 0
    
    chest_pain_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}
    chest_pain_encoded = chest_pain_map[chest_pain]
    
    fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
    
    rest_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    rest_ecg_encoded = rest_ecg_map[rest_ecg]
    
    exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
    
    st_slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
    st_slope_encoded = st_slope_map[st_slope]
    
    thallium_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
    thallium_encoded = thallium_map[thallium]
    
    # Create input array (must match training feature order)
    input_data = np.array([[
        age, sex_encoded, chest_pain_encoded, bp, cholesterol, 
        fasting_bs_encoded, rest_ecg_encoded, max_hr, 
        exercise_angina_encoded, st_depression, st_slope_encoded, 
        num_vessels, thallium_encoded
    ]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 1:
            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
            st.error("### ⚠️ HIGH RISK")
            st.markdown(f"""
            **Risk Probability:** {probability[1]*100:.1f}%
            
            The model predicts a **high risk** of heart disease.
            
            **Recommended Actions:**
            - Consult a cardiologist immediately
            - Schedule comprehensive cardiac evaluation
            - Discuss lifestyle modifications
            - Review family history with your doctor
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
            st.success("### ✅ LOW RISK")
            st.markdown(f"""
            **Risk Probability:** {probability[1]*100:.1f}%
            
            The model predicts a **low risk** of heart disease.
            
            **Recommended Actions:**
            - Maintain healthy lifestyle habits
            - Regular check-ups with your doctor
            - Continue monitoring cardiovascular health
            - Stay physically active
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_result2:
        st.markdown("### 📊 Risk Breakdown")
        st.progress(probability[1])
        st.metric("Disease Risk", f"{probability[1]*100:.1f}%")
        st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
        
        st.markdown("### 📋 Input Summary")
        st.write(f"**Age:** {age} years")
        st.write(f"**Cholesterol:** {cholesterol} mg/dl")
        st.write(f"**Max Heart Rate:** {max_hr} bpm")
        st.write(f"**Blood Pressure:** {bp} mm Hg")
    
    # Download report button
    st.markdown("---")
    report_data = {
        "Prediction Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Sex": sex,
        "Chest Pain Type": chest_pain,
        "Blood Pressure": bp,
        "Cholesterol": cholesterol,
        "Fasting Blood Sugar": fasting_bs,
        "Resting ECG": rest_ecg,
        "Max Heart Rate": max_hr,
        "Exercise Angina": exercise_angina,
        "ST Depression": st_depression,
        "ST Slope": st_slope,
        "Number of Vessels": num_vessels,
        "Thallium": thallium,
        "Prediction": "High Risk" if prediction == 1 else "Low Risk",
        "Disease Probability": f"{probability[1]*100:.1f}%"
    }
    
    report_df = pd.DataFrame([report_data]).T
    report_df.columns = ["Value"]
    
    csv = report_df.to_csv()
    st.download_button(
        label="📥 Download Report (CSV)",
        data=csv,
        file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>Heart Disease Risk Predictor | Northeastern University | MS Information Systems</p>
        <p>Model Accuracy: 88% | Developed using Scikit-learn & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
