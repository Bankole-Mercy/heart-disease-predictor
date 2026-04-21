import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Chatbot",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .bot-message {
        background-color: #1e3a5f;
        border-left: 5px solid #4a90e2;
        color: #ffffff;
    }
    .bot-message strong {
        color: #4a90e2;
    }
    .user-message {
        background-color: #2d5016;
        border-left: 5px solid #66bb6a;
        color: #ffffff;
    }
    .user-message strong {
        color: #66bb6a;
    }
    .risk-high {
        background-color: #8b0000;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        color: #ffffff;
    }
    .risk-low {
        background-color: #1b5e20;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        color: #ffffff;
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

# Initialize session state
if 'conversation_stage' not in st.session_state:
    st.session_state.conversation_stage = 0
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

# Define questions
questions = [
    {
        "key": "age",
        "question": "👋 Hello! I'm your Heart Health Assistant. I'll ask you a few questions to assess your cardiovascular risk.\n\nLet's start: **How old are you?**",
        "type": "number",
        "min": 1,
        "max": 120
    },
    {
        "key": "sex",
        "question": "Thanks! Next question: **What is your biological sex?**",
        "type": "select",
        "options": ["Male", "Female"]
    },
    {
        "key": "chest_pain",
        "question": "Have you experienced any chest pain? If so, **which type best describes it?**",
        "type": "select",
        "options": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    },
    {
        "key": "bp",
        "question": "What is your **resting blood pressure** (in mm Hg)?\n\n💡 *Normal is around 120. If you don't know, you can estimate.*",
        "type": "number",
        "min": 80,
        "max": 220
    },
    {
        "key": "cholesterol",
        "question": "What is your **cholesterol level** (in mg/dl)?\n\n💡 *Normal is around 200. If you don't know, you can estimate.*",
        "type": "number",
        "min": 100,
        "max": 600
    },
    {
        "key": "fasting_bs",
        "question": "Is your **fasting blood sugar greater than 120 mg/dl?**",
        "type": "select",
        "options": ["No", "Yes"]
    },
    {
        "key": "rest_ecg",
        "question": "What were your **resting ECG results?**\n\n💡 *If you haven't had an ECG, select 'Normal'.*",
        "type": "select",
        "options": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    },
    {
        "key": "max_hr",
        "question": "What is your **maximum heart rate** achieved during exercise?\n\n💡 *Normal is 150-180 depending on age.*",
        "type": "number",
        "min": 60,
        "max": 220
    },
    {
        "key": "exercise_angina",
        "question": "Do you experience **chest pain during exercise?**",
        "type": "select",
        "options": ["No", "Yes"]
    },
    {
        "key": "st_depression",
        "question": "What is your **ST depression** value?\n\n💡 *This is from an ECG test. If you don't know, enter 0.*",
        "type": "number",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1
    },
    {
        "key": "st_slope",
        "question": "What is the **slope of your ST segment?**\n\n💡 *From ECG. If unsure, select 'Upsloping'.*",
        "type": "select",
        "options": ["Upsloping", "Flat", "Downsloping"]
    },
    {
        "key": "num_vessels",
        "question": "How many **major vessels** (0-3) are colored by fluoroscopy?\n\n💡 *This is from an angiogram. If you haven't had one, select 0.*",
        "type": "select",
        "options": [0, 1, 2, 3]
    },
    {
        "key": "thallium",
        "question": "What were your **thallium stress test** results?\n\n💡 *If you haven't had this test, select 'Normal'.*",
        "type": "select",
        "options": ["Normal", "Fixed Defect", "Reversible Defect"]
    }
]

# Title
st.markdown('<h1 class="main-header">❤️ Heart Health Chatbot</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Cardiovascular Risk Assessment")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 About This Chatbot")
    st.info("""
    This AI chatbot asks you health questions to predict your heart disease risk.
    
    **Model Performance:**
    - Accuracy: 88%
    - Precision: 88%
    - Recall: 86%
    - F1-Score: 87%
    
    **Progress:** {}/13 questions answered
    """.format(len(st.session_state.patient_data)))
    
    st.header("⚕️ Disclaimer")
    st.warning("""
    This tool is for educational purposes only. 
    It does NOT replace professional medical advice.
    Please consult a healthcare provider.
    """)
    
    if st.button("🔄 Start Over"):
        st.session_state.conversation_stage = 0
        st.session_state.patient_data = {}
        st.session_state.chat_history = []
        st.session_state.show_prediction = False
        st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'bot':
        st.markdown(f'<div class="chat-message bot-message">🤖 <strong>Health Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

# Show current question
if st.session_state.conversation_stage < len(questions) and not st.session_state.show_prediction:
    current_q = questions[st.session_state.conversation_stage]
    
    # Display bot question
    st.markdown(f'<div class="chat-message bot-message">🤖 <strong>Health Assistant:</strong><br>{current_q["question"]}</div>', unsafe_allow_html=True)
    
    # Get user input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if current_q["type"] == "number":
            user_input = st.number_input(
                "Your answer:",
                min_value=current_q["min"],
                max_value=current_q["max"],
                step=current_q.get("step", 1),
                key=f"input_{st.session_state.conversation_stage}"
            )
        else:  # select
            user_input = st.selectbox(
                "Your answer:",
                options=current_q["options"],
                key=f"input_{st.session_state.conversation_stage}"
            )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Submit ➡️", type="primary", use_container_width=True):
            # Save to chat history
            st.session_state.chat_history.append({
                "role": "bot",
                "content": current_q["question"]
            })
            st.session_state.chat_history.append({
                "role": "user",
                "content": str(user_input)
            })
            
            # Save answer
            st.session_state.patient_data[current_q["key"]] = user_input
            
            # Move to next question
            st.session_state.conversation_stage += 1
            
            # Check if we're done
            if st.session_state.conversation_stage >= len(questions):
                st.session_state.show_prediction = True
            
            st.rerun()

# Show prediction if all questions answered
if st.session_state.show_prediction:
    st.markdown('<div class="chat-message bot-message">🤖 <strong>Health Assistant:</strong><br>Thank you for answering all the questions! Let me analyze your responses and calculate your heart disease risk...</div>', unsafe_allow_html=True)
    
    # Prepare data for prediction
    data = st.session_state.patient_data
    
    # Encode categorical variables
    sex_encoded = 1 if data['sex'] == "Male" else 0
    chest_pain_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}
    chest_pain_encoded = chest_pain_map[data['chest_pain']]
    fasting_bs_encoded = 1 if data['fasting_bs'] == "Yes" else 0
    rest_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    rest_ecg_encoded = rest_ecg_map[data['rest_ecg']]
    exercise_angina_encoded = 1 if data['exercise_angina'] == "Yes" else 0
    st_slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
    st_slope_encoded = st_slope_map[data['st_slope']]
    thallium_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
    thallium_encoded = thallium_map[data['thallium']]
    
    # Create input array
    input_data = np.array([[
        data['age'], sex_encoded, chest_pain_encoded, data['bp'], 
        data['cholesterol'], fasting_bs_encoded, rest_ecg_encoded, 
        data['max_hr'], exercise_angina_encoded, data['st_depression'], 
        st_slope_encoded, data['num_vessels'], thallium_encoded
    ]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    
    # Display results
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 1:
            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
            st.error("### ⚠️ HIGH RISK DETECTED")
            st.markdown(f"""
            **Risk Probability:** {probability[1]*100:.1f}%
            
            Based on your responses, the AI model predicts a **high risk** of heart disease.
            
            **⚠️ Important Next Steps:**
            - 🏥 **Consult a cardiologist immediately**
            - 📋 Schedule comprehensive cardiac evaluation
            - 💊 Discuss lifestyle modifications with your doctor
            - 👨‍👩‍👧‍👦 Review family history with healthcare provider
            
            **This is NOT a diagnosis** - only a medical professional can diagnose heart disease.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
            st.success("### ✅ LOW RISK")
            st.markdown(f"""
            **Risk Probability:** {probability[1]*100:.1f}%
            
            Good news! The AI model predicts a **low risk** of heart disease based on your responses.
            
            **💚 Recommended Actions:**
            - 🏃‍♀️ Maintain healthy lifestyle habits
            - 🩺 Continue regular check-ups with your doctor
            - 📊 Monitor your cardiovascular health
            - 💪 Stay physically active
            
            Remember to maintain your heart health through diet, exercise, and regular medical check-ups!
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_result2:
        st.markdown("### 📊 Risk Analysis")
        st.progress(probability[1])
        st.metric("Disease Risk", f"{probability[1]*100:.1f}%", delta=None)
        st.metric("Healthy Probability", f"{probability[0]*100:.1f}%", delta=None)
        
        st.markdown("### 📋 Your Health Summary")
        st.write(f"**Age:** {data['age']} years")
        st.write(f"**Sex:** {data['sex']}")
        st.write(f"**Blood Pressure:** {data['bp']} mm Hg")
        st.write(f"**Cholesterol:** {data['cholesterol']} mg/dl")
        st.write(f"**Max Heart Rate:** {data['max_hr']} bpm")
        st.write(f"**Chest Pain:** {data['chest_pain']}")
    
    # Download report
    st.markdown("---")
    report_data = {
        "Assessment Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **st.session_state.patient_data,
        "Prediction": "High Risk" if prediction == 1 else "Low Risk",
        "Disease Probability": f"{probability[1]*100:.1f}%",
        "Healthy Probability": f"{probability[0]*100:.1f}%"
    }
    
    report_df = pd.DataFrame([report_data]).T
    report_df.columns = ["Value"]
    csv = report_df.to_csv()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv,
            file_name=f"heart_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_btn2:
        if st.button("🔄 Start New Assessment", type="primary", use_container_width=True):
            st.session_state.conversation_stage = 0
            st.session_state.patient_data = {}
            st.session_state.chat_history = []
            st.session_state.show_prediction = False
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #ffffff; padding: 20px;'>
        <p>Heart Disease Risk Chatbot | Northeastern University | MS Information Systems</p>
        <p>Model Accuracy: 88% | Developed by Mercy Bankole</p>
    </div>
""", unsafe_allow_html=True)
