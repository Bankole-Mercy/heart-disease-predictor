"""
Heart Disease Risk Assessment Chatbot
Author: Aanuoluwapo Mercy Bankole
Northeastern University - MS Information Systems
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Heart Health Assistant",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>

/* ── PAGE BACKGROUND ── */
.stApp {
    background-color: #2E3A6E;
}

/* ── MAIN CONTENT CARD ── */
.main .block-container {
    background-color: #3D4E8A;
    border-radius: 20px;
    padding: 0rem 2.5rem 2rem 2.5rem;
    max-width: 860px;
    margin: 1.5rem auto;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.40);
}

/* ── HEADER BANNER ── */
.header-banner {
    background-color: #2A3A7A;
    border-radius: 16px 16px 0 0;
    padding: 1.8rem 2rem 1.4rem 2rem;
    margin: 0 -2.5rem 1.5rem -2.5rem;
    border-bottom: 2px solid #3730A3;
}
.main-header {
    font-size: 2.4rem;
    font-weight: 800;
    color: #FC8181;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.sub-header {
    text-align: center;
    color: #A5B4FC;
    font-size: 1rem;
    margin-bottom: 0.2rem;
}
.author-label {
    text-align: center;
    color: #6366F1;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
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
    background-color: #4A5BA0;
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
    background-color: #3D4E8A;
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
    background-color: #3D4E8A !important;
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
    background-color: #3D4E8A !important;
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
    st.caption("Disclaimer: This is NOT medical advice. Always consult a healthcare professional.")
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
        if q["key"] == "ST depression":
            val = st.number_input(
                label=q["label"],
                min_value=float(q["min"]),
                max_value=float(q["max"]),
                value=float(q["min"]),
                step=0.1,
                format="%.1f",
                label_visibility="collapsed"
            )
            answer_display = str(round(val, 1))
            answer_value = val
        else:
            val = st.number_input(
                label=q["label"],
                min_value=int(q["min"]),
                max_value=int(q["max"]),
                value=int(q["min"]),
                step=1,
                label_visibility="collapsed"
            )
            answer_display = str(int(val))
            answer_value = int(val)
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
    st.caption("Disclaimer: This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")    letter-spacing: 0.05em;
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
    background-color: #4A5BA0;
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
    background-color: #3D4E8A;
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
    background-color: #3D4E8A !important;
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
    background-color: #3D4E8A !important;
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
        if q["key"] == "ST depression":
            val = st.number_input(
                label=q["label"],
                min_value=float(q["min"]),
                max_value=float(q["max"]),
                value=float(q["min"]),
                step=0.1,
                format="%.1f",
                label_visibility="collapsed"
            )
            answer_display = str(round(val, 1))
            answer_value = val
        else:
            val = st.number_input(
                label=q["label"],
                min_value=int(q["min"]),
                max_value=int(q["max"]),
                value=int(q["min"]),
                step=1,
                label_visibility="collapsed"
            )
            answer_display = str(int(val))
            answer_value = int(val)
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
    st.markdown("*Disclaimer: This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*")
