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
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>Heart Disease Risk Chatbot | Northeastern University | MS Information Systems</p>
        <p>Model Accuracy: 88% | Developed by Mercy Bankole</p>
    </div>
""", unsafe_allow_html=True)    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── BOT MESSAGE BUBBLE ── */
.bot-message {
    background-color: #EEF2FF;
    border-left: 5px solid #6366F1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.bot-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── USER MESSAGE BUBBLE ── */
.user-message {
    background-color: #F0FFF4;
    border-left: 5px solid #38A169;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.user-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #38A169;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── RISK RESULT BOXES ── */
.risk-high {
    background-color: #FFF5F5;
    border: 2px solid #FC8181;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #742A2A;
    margin-top: 1rem;
}
.risk-high .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #C53030;
    margin-bottom: 0.5rem;
}
.risk-low {
    background-color: #F0FFF4;
    border: 2px solid #68D391;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #1C4532;
    margin-top: 1rem;
}
.risk-low .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #276749;
    margin-bottom: 0.5rem;
}

/* ── PROGRESS BAR AREA ── */
.progress-label {
    font-size: 0.85rem;
    color: #A5B4FC;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* ── QUESTION CARD ── */
.question-card {
    background-color: #2D2A6E;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid #3730A3;
}
.question-card h3 {
    color: #E0E7FF !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}
.question-hint {
    color: #A5B4FC;
    font-size: 0.83rem;
    margin-bottom: 0.6rem;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1.5px solid #3730A3;
    margin: 1.2rem 0;
}

/* ── ALL TEXT ON DARK BG ── */
.main .block-container p,
.main .block-container label,
.main .block-container span,
.main .block-container div {
    color: #E0E7FF;
}
h3 {
    color: #E0E7FF !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #1E1B4B;
}
section[data-testid="stSidebar"] * {
    color: #E0E7FF !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #A5B4FC !important;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background-color: #6366F1 !important;
}

/* ── INPUT FIELDS ── */
.stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1.5px solid #4F46E5 !important;
    background-color: #2D2A6E !important;
    color: #E0E7FF !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
}

/* ── PROGRESS BAR FILL ── */
.stProgress > div > div {
    background-color: #6366F1 !important;
}
.stProgress > div {
    background-color: #2D2A6E !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background-color: #6366F1 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #4F46E5 !important;
}

</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        return None, None

model, scaler = load_model()


# ── SESSION STATE ────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'started' not in st.session_state:
    st.session_state.started = False


# ── QUESTIONS ────────────────────────────────────────────────────────
QUESTIONS = [
    {"key": "Age",             "label": "How old are you?",                                       "type": "number", "min": 20, "max": 100, "hint": "Enter your age in years"},
    {"key": "Sex",             "label": "What is your biological sex?",                           "type": "select", "options": {"Male": 1, "Female": 0}},
    {"key": "Chest pain type", "label": "Which best describes your chest pain?",                  "type": "select", "options": {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic (no pain)": 4}},
    {"key": "BP",              "label": "What is your resting blood pressure? (mm Hg)",           "type": "number", "min": 80, "max": 220, "hint": "Normal range: 90–140 mm Hg"},
    {"key": "Cholesterol",     "label": "What is your cholesterol level? (mg/dl)",                "type": "number", "min": 100, "max": 600, "hint": "Normal range: 125–200 mg/dl"},
    {"key": "FBS over 120",    "label": "Is your fasting blood sugar above 120 mg/dl?",           "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "EKG results",     "label": "What are your resting EKG results?",                     "type": "select", "options": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
    {"key": "Max HR",          "label": "What is your maximum heart rate achieved?",              "type": "number", "min": 60, "max": 220, "hint": "Typical range: 100–200 bpm"},
    {"key": "Exercise angina", "label": "Do you experience chest pain during exercise?",          "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "ST depression",   "label": "What is your ST depression level? (exercise vs rest)",   "type": "number", "min": 0, "max": 10, "hint": "Enter a value between 0.0 and 6.2"},
    {"key": "Slope of ST",     "label": "What is the slope of your peak exercise ST segment?",    "type": "select", "options": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}},
    {"key": "Number of vessels fluro", "label": "How many major vessels are coloured by fluoroscopy?", "type": "select", "options": {"0": 0, "1": 1, "2": 2, "3": 3}},
    {"key": "Thallium",        "label": "What is your thallium stress test result?",              "type": "select", "options": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}},
]

TOTAL = len(QUESTIONS)


# ── HELPERS ──────────────────────────────────────────────────────────
def bot_bubble(text):
    st.markdown(f"""
        <div class="bot-message">
            <div class="sender-label">❤️ Heart Assistant</div>
            {text}
        </div>""", unsafe_allow_html=True)

def user_bubble(text):
    st.markdown(f"""
        <div class="user-message">
            <div class="sender-label">👤 You</div>
            {text}
        </div>""", unsafe_allow_html=True)

def make_prediction(data):
    try:
        input_df = pd.DataFrame([data])
        num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
        existing = [c for c in num_cols if c in input_df.columns]
        input_df[existing] = scaler.transform(input_df[existing])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        return pred, prob
    except Exception as e:
        return None, str(e)


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Assistant")
    st.markdown("---")

    if st.session_state.step > 0:
        pct = int((st.session_state.step / TOTAL) * 100)
        st.markdown(f"**Progress: {st.session_state.step}/{TOTAL} questions**")
        st.progress(pct)
    else:
        st.markdown("**Progress: Not started**")
        st.progress(0)

    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown("This chatbot collects 13 clinical measurements and uses a trained Logistic Regression model to assess heart disease risk.")
    st.markdown("---")
    st.markdown("*Disclaimer: This is NOT medical advice. Always consult a healthcare professional.*")
    st.markdown("---")

    if st.button("🔄 Start Over"):
        st.session_state.messages = []
        st.session_state.step = 0
        st.session_state.user_data = {}
        st.session_state.started = False
        st.rerun()


# ── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1 class="main-header">❤️ Heart Health Assistant</h1>
    <p class="sub-header">Answer 13 simple questions to receive your heart disease risk assessment</p>
    <p class="author-label">Aanuoluwapo Mercy Bankole &nbsp;·&nbsp; MS Information Systems &nbsp;·&nbsp; Northeastern University</p>
</div>
""", unsafe_allow_html=True)


# ── CHAT HISTORY ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        bot_bubble(msg["content"])
    else:
        user_bubble(msg["content"])


# ── CONVERSATION FLOW ─────────────────────────────────────────────────
if not st.session_state.started:
    bot_bubble("👋 Hello! I'm your Heart Health Assistant.<br><br>I'll ask you <strong>13 short questions</strong> about your health. Based on your answers, I'll assess your heart disease risk using a machine learning model.<br><br>Click <strong>Start</strong> below when you're ready.")
    if st.button("Start Assessment ▶"):
        st.session_state.started = True
        st.session_state.messages.append({"role": "bot", "content": "👋 Hello! I'm your Heart Health Assistant. Let's begin!"})
        st.rerun()

elif st.session_state.step < TOTAL:
    q = QUESTIONS[st.session_state.step]
    step_num = st.session_state.step + 1

    st.markdown(f'<p class="progress-label">Question {step_num} of {TOTAL}</p>', unsafe_allow_html=True)
    st.progress(int((step_num / TOTAL) * 100))

    st.markdown(f'<div class="question-card"><h3>{q["label"]}</h3>', unsafe_allow_html=True)
    if q["type"] == "number":
        st.markdown(f'<p class="question-hint">{q.get("hint", "")}</p>', unsafe_allow_html=True)
        val = st.number_input(
            label=q["label"],
            min_value=float(q["min"]),
            max_value=float(q["max"]),
            value=float(q["min"]),
            label_visibility="collapsed"
        )
        answer_display = str(val)
        answer_value = val
    else:
        option_labels = list(q["options"].keys())
        choice = st.selectbox(
            label=q["label"],
            options=option_labels,
            label_visibility="collapsed"
        )
        answer_display = choice
        answer_value = q["options"][choice]
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f"Next ➜", key=f"next_{st.session_state.step}"):
        st.session_state.messages.append({"role": "bot",  "content": f"<strong>Question {step_num}/{TOTAL}:</strong> {q['label']}"})
        st.session_state.messages.append({"role": "user", "content": answer_display})
        st.session_state.user_data[q["key"]] = answer_value
        st.session_state.step += 1
        st.rerun()

else:
    # ── PREDICTION ───────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    bot_bubble("✅ All questions answered! Analysing your data now...")

    if model is not None:
        pred, prob = make_prediction(st.session_state.user_data)

        if pred is not None:
            confidence = round(prob[pred] * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-title">⚠️ Higher Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>higher likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Please consult a qualified healthcare professional for a full medical evaluation. Early detection saves lives.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-title">✅ Lower Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>lower likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Keep up healthy habits — regular exercise, balanced diet, and routine check-ups are key.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your answers summary:**")
            summary_df = pd.DataFrame(
                list(st.session_state.user_data.items()),
                columns=["Feature", "Your Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Prediction error: {prob}")
    else:
        st.warning("Model not loaded. Please check that heart_disease_model.pkl and scaler.pkl are in the same folder as app.py.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("*Disclaimer: This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*")    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── BOT MESSAGE BUBBLE ── */
.bot-message {
    background-color: #EEF2FF;
    border-left: 5px solid #6366F1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.bot-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── USER MESSAGE BUBBLE ── */
.user-message {
    background-color: #F0FFF4;
    border-left: 5px solid #38A169;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.user-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #38A169;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── RISK RESULT BOXES ── */
.risk-high {
    background-color: #FFF5F5;
    border: 2px solid #FC8181;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #742A2A;
    margin-top: 1rem;
}
.risk-high .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #C53030;
    margin-bottom: 0.5rem;
}
.risk-low {
    background-color: #F0FFF4;
    border: 2px solid #68D391;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #1C4532;
    margin-top: 1rem;
}
.risk-low .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #276749;
    margin-bottom: 0.5rem;
}

/* ── PROGRESS BAR AREA ── */
.progress-label {
    font-size: 0.85rem;
    color: #A5B4FC;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* ── QUESTION CARD ── */
.question-card {
    background-color: #2D2A6E;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid #3730A3;
}
.question-card h3 {
    color: #E0E7FF !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}
.question-hint {
    color: #A5B4FC;
    font-size: 0.83rem;
    margin-bottom: 0.6rem;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1.5px solid #3730A3;
    margin: 1.2rem 0;
}

/* ── ALL TEXT ON DARK BG ── */
.main .block-container p,
.main .block-container label,
.main .block-container span,
.main .block-container div {
    color: #E0E7FF;
}
h3 {
    color: #E0E7FF !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #1E1B4B;
}
section[data-testid="stSidebar"] * {
    color: #E0E7FF !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #A5B4FC !important;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background-color: #6366F1 !important;
}

/* ── INPUT FIELDS ── */
.stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1.5px solid #4F46E5 !important;
    background-color: #2D2A6E !important;
    color: #E0E7FF !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
}

/* ── PROGRESS BAR FILL ── */
.stProgress > div > div {
    background-color: #6366F1 !important;
}
.stProgress > div {
    background-color: #2D2A6E !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background-color: #6366F1 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #4F46E5 !important;
}

</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        return None, None

model, scaler = load_model()


# ── SESSION STATE ────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'started' not in st.session_state:
    st.session_state.started = False


# ── QUESTIONS ────────────────────────────────────────────────────────
QUESTIONS = [
    {"key": "Age",             "label": "How old are you?",                                       "type": "number", "min": 20, "max": 100, "hint": "Enter your age in years"},
    {"key": "Sex",             "label": "What is your biological sex?",                           "type": "select", "options": {"Male": 1, "Female": 0}},
    {"key": "Chest pain type", "label": "Which best describes your chest pain?",                  "type": "select", "options": {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic (no pain)": 4}},
    {"key": "BP",              "label": "What is your resting blood pressure? (mm Hg)",           "type": "number", "min": 80, "max": 220, "hint": "Normal range: 90–140 mm Hg"},
    {"key": "Cholesterol",     "label": "What is your cholesterol level? (mg/dl)",                "type": "number", "min": 100, "max": 600, "hint": "Normal range: 125–200 mg/dl"},
    {"key": "FBS over 120",    "label": "Is your fasting blood sugar above 120 mg/dl?",           "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "EKG results",     "label": "What are your resting EKG results?",                     "type": "select", "options": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
    {"key": "Max HR",          "label": "What is your maximum heart rate achieved?",              "type": "number", "min": 60, "max": 220, "hint": "Typical range: 100–200 bpm"},
    {"key": "Exercise angina", "label": "Do you experience chest pain during exercise?",          "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "ST depression",   "label": "What is your ST depression level? (exercise vs rest)",   "type": "number", "min": 0, "max": 10, "hint": "Enter a value between 0.0 and 6.2"},
    {"key": "Slope of ST",     "label": "What is the slope of your peak exercise ST segment?",    "type": "select", "options": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}},
    {"key": "Number of vessels fluro", "label": "How many major vessels are coloured by fluoroscopy?", "type": "select", "options": {"0": 0, "1": 1, "2": 2, "3": 3}},
    {"key": "Thallium",        "label": "What is your thallium stress test result?",              "type": "select", "options": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}},
]

TOTAL = len(QUESTIONS)


# ── HELPERS ──────────────────────────────────────────────────────────
def bot_bubble(text):
    st.markdown(f"""
        <div class="bot-message">
            <div class="sender-label">❤️ Heart Assistant</div>
            {text}
        </div>""", unsafe_allow_html=True)

def user_bubble(text):
    st.markdown(f"""
        <div class="user-message">
            <div class="sender-label">👤 You</div>
            {text}
        </div>""", unsafe_allow_html=True)

def make_prediction(data):
    try:
        input_df = pd.DataFrame([data])
        num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
        existing = [c for c in num_cols if c in input_df.columns]
        input_df[existing] = scaler.transform(input_df[existing])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        return pred, prob
    except Exception as e:
        return None, str(e)


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Assistant")
    st.markdown("---")

    if st.session_state.step > 0:
        pct = int((st.session_state.step / TOTAL) * 100)
        st.markdown(f"**Progress: {st.session_state.step}/{TOTAL} questions**")
        st.progress(pct)
    else:
        st.markdown("**Progress: Not started**")
        st.progress(0)

    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown("This chatbot collects 13 clinical measurements and uses a trained Logistic Regression model to assess heart disease risk.")
    st.markdown("---")
    st.markdown("⚠️ *This is NOT medical advice. Always consult a healthcare professional.*")
    st.markdown("---")

    if st.button("🔄 Start Over"):
        st.session_state.messages = []
        st.session_state.step = 0
        st.session_state.user_data = {}
        st.session_state.started = False
        st.rerun()


# ── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1 class="main-header">❤️ Heart Health Assistant</h1>
    <p class="sub-header">Answer 13 simple questions to receive your heart disease risk assessment</p>
    <p class="author-label">Aanuoluwapo Mercy Bankole &nbsp;·&nbsp; MS Information Systems &nbsp;·&nbsp; Northeastern University</p>
</div>
""", unsafe_allow_html=True)


# ── CHAT HISTORY ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        bot_bubble(msg["content"])
    else:
        user_bubble(msg["content"])


# ── CONVERSATION FLOW ─────────────────────────────────────────────────
if not st.session_state.started:
    bot_bubble("👋 Hello! I'm your Heart Health Assistant.<br><br>I'll ask you <strong>13 short questions</strong> about your health. Based on your answers, I'll assess your heart disease risk using a machine learning model.<br><br>Click <strong>Start</strong> below when you're ready.")
    if st.button("Start Assessment ▶"):
        st.session_state.started = True
        st.session_state.messages.append({"role": "bot", "content": "👋 Hello! I'm your Heart Health Assistant. Let's begin!"})
        st.rerun()

elif st.session_state.step < TOTAL:
    q = QUESTIONS[st.session_state.step]
    step_num = st.session_state.step + 1

    st.markdown(f'<p class="progress-label">Question {step_num} of {TOTAL}</p>', unsafe_allow_html=True)
    st.progress(int((step_num / TOTAL) * 100))

    st.markdown(f'<div class="question-card"><h3>{q["label"]}</h3>', unsafe_allow_html=True)
    if q["type"] == "number":
        st.markdown(f'<p class="question-hint">{q.get("hint", "")}</p>', unsafe_allow_html=True)
        val = st.number_input(
            label=q["label"],
            min_value=float(q["min"]),
            max_value=float(q["max"]),
            value=float(q["min"]),
            label_visibility="collapsed"
        )
        answer_display = str(val)
        answer_value = val
    else:
        option_labels = list(q["options"].keys())
        choice = st.selectbox(
            label=q["label"],
            options=option_labels,
            label_visibility="collapsed"
        )
        answer_display = choice
        answer_value = q["options"][choice]
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f"Next ➜", key=f"next_{st.session_state.step}"):
        st.session_state.messages.append({"role": "bot",  "content": f"<strong>Question {step_num}/{TOTAL}:</strong> {q['label']}"})
        st.session_state.messages.append({"role": "user", "content": answer_display})
        st.session_state.user_data[q["key"]] = answer_value
        st.session_state.step += 1
        st.rerun()

else:
    # ── PREDICTION ───────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    bot_bubble("✅ All questions answered! Analysing your data now...")

    if model is not None:
        pred, prob = make_prediction(st.session_state.user_data)

        if pred is not None:
            confidence = round(prob[pred] * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-title">⚠️ Higher Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>higher likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Please consult a qualified healthcare professional for a full medical evaluation. Early detection saves lives.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-title">✅ Lower Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>lower likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Keep up healthy habits — regular exercise, balanced diet, and routine check-ups are key.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your answers summary:**")
            summary_df = pd.DataFrame(
                list(st.session_state.user_data.items()),
                columns=["Feature", "Your Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Prediction error: {prob}")
    else:
        st.warning("Model not loaded. Please check that heart_disease_model.pkl and scaler.pkl are in the same folder as app.py.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("⚠️ *This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*")    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── BOT MESSAGE BUBBLE ── */
.bot-message {
    background-color: #EEF2FF;
    border-left: 5px solid #6366F1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.bot-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── USER MESSAGE BUBBLE ── */
.user-message {
    background-color: #F0FFF4;
    border-left: 5px solid #38A169;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.user-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #38A169;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── RISK RESULT BOXES ── */
.risk-high {
    background-color: #FFF5F5;
    border: 2px solid #FC8181;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #742A2A;
    margin-top: 1rem;
}
.risk-high .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #C53030;
    margin-bottom: 0.5rem;
}
.risk-low {
    background-color: #F0FFF4;
    border: 2px solid #68D391;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #1C4532;
    margin-top: 1rem;
}
.risk-low .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #276749;
    margin-bottom: 0.5rem;
}

/* ── PROGRESS BAR AREA ── */
.progress-label {
    font-size: 0.85rem;
    color: #A5B4FC;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* ── QUESTION CARD ── */
.question-card {
    background-color: #2D2A6E;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid #3730A3;
}
.question-card h3 {
    color: #E0E7FF !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}
.question-hint {
    color: #A5B4FC;
    font-size: 0.83rem;
    margin-bottom: 0.6rem;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1.5px solid #3730A3;
    margin: 1.2rem 0;
}

/* ── ALL TEXT ON DARK BG ── */
.main .block-container p,
.main .block-container label,
.main .block-container span,
.main .block-container div {
    color: #E0E7FF;
}
h3 {
    color: #E0E7FF !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #1E1B4B;
}
section[data-testid="stSidebar"] * {
    color: #E0E7FF !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #A5B4FC !important;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background-color: #6366F1 !important;
}

/* ── INPUT FIELDS ── */
.stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1.5px solid #4F46E5 !important;
    background-color: #2D2A6E !important;
    color: #E0E7FF !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
}

/* ── PROGRESS BAR FILL ── */
.stProgress > div > div {
    background-color: #6366F1 !important;
}
.stProgress > div {
    background-color: #2D2A6E !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background-color: #6366F1 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #4F46E5 !important;
}

</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        return None, None

model, scaler = load_model()


# ── SESSION STATE ────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'started' not in st.session_state:
    st.session_state.started = False


# ── QUESTIONS ────────────────────────────────────────────────────────
QUESTIONS = [
    {"key": "Age",             "label": "How old are you?",                                       "type": "number", "min": 20, "max": 100, "hint": "Enter your age in years"},
    {"key": "Sex",             "label": "What is your biological sex?",                           "type": "select", "options": {"Male": 1, "Female": 0}},
    {"key": "Chest pain type", "label": "Which best describes your chest pain?",                  "type": "select", "options": {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic (no pain)": 4}},
    {"key": "BP",              "label": "What is your resting blood pressure? (mm Hg)",           "type": "number", "min": 80, "max": 220, "hint": "Normal range: 90–140 mm Hg"},
    {"key": "Cholesterol",     "label": "What is your cholesterol level? (mg/dl)",                "type": "number", "min": 100, "max": 600, "hint": "Normal range: 125–200 mg/dl"},
    {"key": "FBS over 120",    "label": "Is your fasting blood sugar above 120 mg/dl?",           "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "EKG results",     "label": "What are your resting EKG results?",                     "type": "select", "options": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
    {"key": "Max HR",          "label": "What is your maximum heart rate achieved?",              "type": "number", "min": 60, "max": 220, "hint": "Typical range: 100–200 bpm"},
    {"key": "Exercise angina", "label": "Do you experience chest pain during exercise?",          "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "ST depression",   "label": "What is your ST depression level? (exercise vs rest)",   "type": "number", "min": 0, "max": 10, "hint": "Enter a value between 0.0 and 6.2"},
    {"key": "Slope of ST",     "label": "What is the slope of your peak exercise ST segment?",    "type": "select", "options": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}},
    {"key": "Number of vessels fluro", "label": "How many major vessels are coloured by fluoroscopy?", "type": "select", "options": {"0": 0, "1": 1, "2": 2, "3": 3}},
    {"key": "Thallium",        "label": "What is your thallium stress test result?",              "type": "select", "options": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}},
]

TOTAL = len(QUESTIONS)


# ── HELPERS ──────────────────────────────────────────────────────────
def bot_bubble(text):
    st.markdown(f"""
        <div class="bot-message">
            <div class="sender-label">❤️ Heart Assistant</div>
            {text}
        </div>""", unsafe_allow_html=True)

def user_bubble(text):
    st.markdown(f"""
        <div class="user-message">
            <div class="sender-label">👤 You</div>
            {text}
        </div>""", unsafe_allow_html=True)

def make_prediction(data):
    try:
        input_df = pd.DataFrame([data])
        num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
        existing = [c for c in num_cols if c in input_df.columns]
        input_df[existing] = scaler.transform(input_df[existing])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        return pred, prob
    except Exception as e:
        return None, str(e)


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Assistant")
    st.markdown("---")

    if st.session_state.step > 0:
        pct = int((st.session_state.step / TOTAL) * 100)
        st.markdown(f"**Progress: {st.session_state.step}/{TOTAL} questions**")
        st.progress(pct)
    else:
        st.markdown("**Progress: Not started**")
        st.progress(0)

    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown("This chatbot collects 13 clinical measurements and uses a trained Logistic Regression model to assess heart disease risk.")
    st.markdown("---")
    st.markdown("⚠️ *This is NOT medical advice. Always consult a healthcare professional.*")
    st.markdown("---")

    if st.button("🔄 Start Over"):
        st.session_state.messages = []
        st.session_state.step = 0
        st.session_state.user_data = {}
        st.session_state.started = False
        st.rerun()


# ── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1 class="main-header">❤️ Heart Health Assistant</h1>
    <p class="sub-header">Answer 13 simple questions to receive your heart disease risk assessment</p>
    <p class="author-label">Aanuoluwapo Mercy Bankole &nbsp;·&nbsp; MS Information Systems &nbsp;·&nbsp; Northeastern University</p>
</div>
""", unsafe_allow_html=True)


# ── CHAT HISTORY ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        bot_bubble(msg["content"])
    else:
        user_bubble(msg["content"])


# ── CONVERSATION FLOW ─────────────────────────────────────────────────
if not st.session_state.started:
    bot_bubble("👋 Hello! I'm your Heart Health Assistant.<br><br>I'll ask you <strong>13 short questions</strong> about your health. Based on your answers, I'll assess your heart disease risk using a machine learning model.<br><br>Click <strong>Start</strong> below when you're ready.")
    if st.button("Start Assessment ▶"):
        st.session_state.started = True
        st.session_state.messages.append({"role": "bot", "content": "👋 Hello! I'm your Heart Health Assistant. Let's begin!"})
        st.rerun()

elif st.session_state.step < TOTAL:
    q = QUESTIONS[st.session_state.step]
    step_num = st.session_state.step + 1

    st.markdown(f'<p class="progress-label">Question {step_num} of {TOTAL}</p>', unsafe_allow_html=True)
    st.progress(int((step_num / TOTAL) * 100))

    st.markdown(f'<div class="question-card"><h3>{q["label"]}</h3>', unsafe_allow_html=True)
    if q["type"] == "number":
        st.markdown(f'<p class="question-hint">{q.get("hint", "")}</p>', unsafe_allow_html=True)
        val = st.number_input(
            label=q["label"],
            min_value=float(q["min"]),
            max_value=float(q["max"]),
            value=float(q["min"]),
            label_visibility="collapsed"
        )
        answer_display = str(val)
        answer_value = val
    else:
        option_labels = list(q["options"].keys())
        choice = st.selectbox(
            label=q["label"],
            options=option_labels,
            label_visibility="collapsed"
        )
        answer_display = choice
        answer_value = q["options"][choice]
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f"Next ➜", key=f"next_{st.session_state.step}"):
        st.session_state.messages.append({"role": "bot",  "content": f"<strong>Question {step_num}/{TOTAL}:</strong> {q['label']}"})
        st.session_state.messages.append({"role": "user", "content": answer_display})
        st.session_state.user_data[q["key"]] = answer_value
        st.session_state.step += 1
        st.rerun()

else:
    # ── PREDICTION ───────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    bot_bubble("✅ All questions answered! Analysing your data now...")

    if model is not None:
        pred, prob = make_prediction(st.session_state.user_data)

        if pred is not None:
            confidence = round(prob[pred] * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-title">⚠️ Higher Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>higher likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Please consult a qualified healthcare professional for a full medical evaluation. Early detection saves lives.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-title">✅ Lower Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>lower likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Keep up healthy habits — regular exercise, balanced diet, and routine check-ups are key.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your answers summary:**")
            summary_df = pd.DataFrame(
                list(st.session_state.user_data.items()),
                columns=["Feature", "Your Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Prediction error: {prob}")
    else:
        st.warning("Model not loaded. Please check that heart_disease_model.pkl and scaler.pkl are in the same folder as app.py.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("⚠️ *This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*").bot-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── USER MESSAGE BUBBLE ── */
.user-message {
    background-color: #F0FFF4;
    border-left: 5px solid #38A169;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #1A202C;
    font-size: 0.97rem;
    line-height: 1.65;
}
.user-message .sender-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #38A169;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* ── RISK RESULT BOXES ── */
.risk-high {
    background-color: #FFF5F5;
    border: 2px solid #FC8181;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #742A2A;
    margin-top: 1rem;
}
.risk-high .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #C53030;
    margin-bottom: 0.5rem;
}
.risk-low {
    background-color: #F0FFF4;
    border: 2px solid #68D391;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #1C4532;
    margin-top: 1rem;
}
.risk-low .risk-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #276749;
    margin-bottom: 0.5rem;
}

/* ── PROGRESS BAR AREA ── */
.progress-label {
    font-size: 0.85rem;
    color: #6366F1;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1.5px solid #E2E8F0;
    margin: 1.2rem 0;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #1E1B4B;
}
section[data-testid="stSidebar"] * {
    color: #E0E7FF !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #A5B4FC !important;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background-color: #6366F1 !important;
}

/* ── INPUT FIELDS ── */
.stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1.5px solid #C7D2FE !important;
    background-color: #F8FAFF !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background-color: #6366F1 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #4F46E5 !important;
}

</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        return None, None

model, scaler = load_model()


# ── SESSION STATE ────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'started' not in st.session_state:
    st.session_state.started = False


# ── QUESTIONS ────────────────────────────────────────────────────────
QUESTIONS = [
    {"key": "Age",             "label": "How old are you?",                                       "type": "number", "min": 20, "max": 100, "hint": "Enter your age in years"},
    {"key": "Sex",             "label": "What is your biological sex?",                           "type": "select", "options": {"Male": 1, "Female": 0}},
    {"key": "Chest pain type", "label": "Which best describes your chest pain?",                  "type": "select", "options": {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic (no pain)": 4}},
    {"key": "BP",              "label": "What is your resting blood pressure? (mm Hg)",           "type": "number", "min": 80, "max": 220, "hint": "Normal range: 90–140 mm Hg"},
    {"key": "Cholesterol",     "label": "What is your cholesterol level? (mg/dl)",                "type": "number", "min": 100, "max": 600, "hint": "Normal range: 125–200 mg/dl"},
    {"key": "FBS over 120",    "label": "Is your fasting blood sugar above 120 mg/dl?",           "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "EKG results",     "label": "What are your resting EKG results?",                     "type": "select", "options": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
    {"key": "Max HR",          "label": "What is your maximum heart rate achieved?",              "type": "number", "min": 60, "max": 220, "hint": "Typical range: 100–200 bpm"},
    {"key": "Exercise angina", "label": "Do you experience chest pain during exercise?",          "type": "select", "options": {"No": 0, "Yes": 1}},
    {"key": "ST depression",   "label": "What is your ST depression level? (exercise vs rest)",   "type": "number", "min": 0, "max": 10, "hint": "Enter a value between 0.0 and 6.2"},
    {"key": "Slope of ST",     "label": "What is the slope of your peak exercise ST segment?",    "type": "select", "options": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}},
    {"key": "Number of vessels fluro", "label": "How many major vessels are coloured by fluoroscopy?", "type": "select", "options": {"0": 0, "1": 1, "2": 2, "3": 3}},
    {"key": "Thallium",        "label": "What is your thallium stress test result?",              "type": "select", "options": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}},
]

TOTAL = len(QUESTIONS)


# ── HELPERS ──────────────────────────────────────────────────────────
def bot_bubble(text):
    st.markdown(f"""
        <div class="bot-message">
            <div class="sender-label">❤️ Heart Assistant</div>
            {text}
        </div>""", unsafe_allow_html=True)

def user_bubble(text):
    st.markdown(f"""
        <div class="user-message">
            <div class="sender-label">👤 You</div>
            {text}
        </div>""", unsafe_allow_html=True)

def make_prediction(data):
    try:
        input_df = pd.DataFrame([data])
        num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
        existing = [c for c in num_cols if c in input_df.columns]
        input_df[existing] = scaler.transform(input_df[existing])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        return pred, prob
    except Exception as e:
        return None, str(e)


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Assistant")
    st.markdown("---")

    if st.session_state.step > 0:
        pct = int((st.session_state.step / TOTAL) * 100)
        st.markdown(f"**Progress: {st.session_state.step}/{TOTAL} questions**")
        st.progress(pct)
    else:
        st.markdown("**Progress: Not started**")
        st.progress(0)

    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown("This chatbot collects 13 clinical measurements and uses a trained Logistic Regression model to assess heart disease risk.")
    st.markdown("---")
    st.markdown("⚠️ *This is NOT medical advice. Always consult a healthcare professional.*")
    st.markdown("---")

    if st.button("🔄 Start Over"):
        st.session_state.messages = []
        st.session_state.step = 0
        st.session_state.user_data = {}
        st.session_state.started = False
        st.rerun()


# ── HEADER ───────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">❤️ Heart Health Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Answer 13 simple questions to receive your heart disease risk assessment</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ── CHAT HISTORY ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        bot_bubble(msg["content"])
    else:
        user_bubble(msg["content"])


# ── CONVERSATION FLOW ─────────────────────────────────────────────────
if not st.session_state.started:
    bot_bubble("👋 Hello! I'm your Heart Health Assistant.<br><br>I'll ask you <strong>13 short questions</strong> about your health. Based on your answers, I'll assess your heart disease risk using a machine learning model.<br><br>Click <strong>Start</strong> below when you're ready.")
    if st.button("Start Assessment ▶"):
        st.session_state.started = True
        st.session_state.messages.append({"role": "bot", "content": "👋 Hello! I'm your Heart Health Assistant. Let's begin!"})
        st.rerun()

elif st.session_state.step < TOTAL:
    q = QUESTIONS[st.session_state.step]
    step_num = st.session_state.step + 1

    st.markdown(f'<p class="progress-label">Question {step_num} of {TOTAL}</p>', unsafe_allow_html=True)
    st.progress(int((step_num / TOTAL) * 100))

    st.markdown(f"### {q['label']}")
    if q["type"] == "number":
        st.caption(q.get("hint", ""))
        val = st.number_input(
            label=q["label"],
            min_value=float(q["min"]),
            max_value=float(q["max"]),
            value=float(q["min"]),
            label_visibility="collapsed"
        )
        answer_display = str(val)
        answer_value = val
    else:
        option_labels = list(q["options"].keys())
        choice = st.selectbox(
            label=q["label"],
            options=option_labels,
            label_visibility="collapsed"
        )
        answer_display = choice
        answer_value = q["options"][choice]

    if st.button(f"Next ➜", key=f"next_{st.session_state.step}"):
        st.session_state.messages.append({"role": "bot",  "content": f"<strong>Question {step_num}/{TOTAL}:</strong> {q['label']}"})
        st.session_state.messages.append({"role": "user", "content": answer_display})
        st.session_state.user_data[q["key"]] = answer_value
        st.session_state.step += 1
        st.rerun()

else:
    # ── PREDICTION ───────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    bot_bubble("✅ All questions answered! Analysing your data now...")

    if model is not None:
        pred, prob = make_prediction(st.session_state.user_data)

        if pred is not None:
            confidence = round(prob[pred] * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-title">⚠️ Higher Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>higher likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Please consult a qualified healthcare professional for a full medical evaluation. Early detection saves lives.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-title">✅ Lower Risk Detected</div>
                    <p>Based on your responses, the model predicts a <strong>lower likelihood of heart disease</strong>.</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p>Keep up healthy habits — regular exercise, balanced diet, and routine check-ups are key.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your answers summary:**")
            summary_df = pd.DataFrame(
                list(st.session_state.user_data.items()),
                columns=["Feature", "Your Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Prediction error: {prob}")
    else:
        st.warning("Model not loaded. Please check that heart_disease_model.pkl and scaler.pkl are in the same folder as app.py.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("⚠️ *This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*")
