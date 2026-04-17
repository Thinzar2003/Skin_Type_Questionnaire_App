import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import joblib
import os

# ── 1. CONFIG & SYSTEM SETUP ──────────────────────────────────────
st.set_page_config(page_title='AI Skin Analyzer Pro', page_icon='🧴', layout='centered')

# Initialize Gemini (Replace with your actual API key)
GEMINI_API_KEY = "AQ.Ab8RN6KfgihTts6mlJXryK_7_o63IRqlAWKzxcVjyqA_84JqtA" 
genai.configure(api_key=GEMINI_API_KEY)

# ── 2. AI MODEL LOGIC (PREDICTIVE) ────────────────────────────────
def get_trained_model():
    """
    Creates a synthetic dataset and trains a Random Forest model.
    In a real research paper, you would use a validated clinical dataset.
    """
    if os.path.exists('skin_model.pkl'):
        return joblib.load('skin_model.pkl')
    
    # Generate Synthetic Academic Data for training (500 samples)
    data = []
    labels = ['Dry', 'Normal', 'Oily', 'Combination']
    for _ in range(500):
        s_type = np.random.choice(labels)
        if s_type == 'Oily': row = [np.random.randint(2,4) for _ in range(8)]
        elif s_type == 'Dry': row = [np.random.randint(0,2) for _ in range(8)]
        else: row = [np.random.randint(0,4) for _ in range(8)]
        data.append(row + [s_type])
    
    df = pd.DataFrame(data, columns=[f'q{i}' for i in range(8)] + ['target'])
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'skin_model.pkl')
    return model

# ── 3. GENERATIVE AI LOGIC (CONSULTATION) ──────────────────────────
def generate_ai_report(skin_type, confidence, answers):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Act as a professional Dermatologist. 
        Analysis Result: {skin_type} skin with {confidence}% model confidence.
        User's specific answers: {answers}.
        
        Provide a detailed report including:
        1. A scientific explanation of why they have this skin type.
        2. A customized 3-step skincare routine.
        3. Ingredients to look for and ingredients to avoid.
        4. A concluding remark in both English and Thai (since the user is in Thailand).
        
        Keep the tone academic but accessible.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Consultation currently unavailable. (Error: {e})"

# ── 4. UI STYLING ──────────────────────────────────────────────────
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stButton>button { background-color: #2d2d2d; color: white; border-radius: 10px; height: 3em; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #4a4a4a; border: 1px solid #2d2d2d; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #2d2d2d; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 20px; }
</style>
''', unsafe_allow_html=True)

# ── 5. APPLICATION UI ─────────────────────────────────────────────
st.title('🧴 AI Skin Type Research Pro')
st.markdown('**Methodology:** This application uses a *Random Forest Classifier* for classification and *Gemini 1.5 Flash* for generative dermatological consultation.')

# Mapping responses to integers for the ML model
VAL_MAP = {
    # Q1
    'Very tight and uncomfortable': 0, 'Slightly tight': 1, 'Comfortable and balanced': 2, 'Fine, no particular feeling': 3,
    # Q2
    'Very shiny all over': 3, 'Shiny only on forehead, nose, chin (T-zone)': 2, 'Still looks the same as morning': 1, 'Feels drier and tighter': 0,
    # Q3
    'Frequently, all over face': 3, 'Occasionally, mainly T-zone': 2, 'Rarely': 1, 'Almost never, but skin feels flaky': 0,
    # Q4
    'Very dry, flaky or itchy': 0, 'Slightly dry in some areas': 1, 'Normal, no issues': 2, 'Gets oily quickly': 3,
    # Q5
    'Large and visible, especially on nose': 3, 'Visible only on T-zone': 2, 'Small and barely visible': 1, 'Very small, skin looks tight': 0,
    # Q6
    'Often gets irritated or red': 0, 'Sometimes breaks out': 2, 'Rarely reacts': 1, 'Absorbs products quickly, needs more': 3,
    # Q7
    'Rough, flaky or tight': 0, 'Smooth in some areas, oily in others': 2, 'Smooth and balanced overall': 1, 'Consistently shiny and greasy': 3,
    # Q8
    'A lot of oil all over': 3, 'Oil mainly from T-zone': 2, 'Very little oil': 1, 'Almost nothing, skin is dry': 0
}

# The Questionnaire
QUESTIONS = [
    ("How does your skin feel 1 hour after washing?", ['Very tight and uncomfortable', 'Slightly tight', 'Comfortable and balanced', 'Fine, no particular feeling']),
    ("By midday, how does your skin look?", ['Very shiny all over', 'Shiny only on forehead, nose, chin (T-zone)', 'Still looks the same as morning', 'Feels drier and tighter']),
    ("How often do you experience breakouts?", ['Frequently, all over face', 'Occasionally, mainly T-zone', 'Rarely', 'Almost never, but skin feels flaky']),
    ("How does skin feel without moisturizer?", ['Very dry, flaky or itchy', 'Slightly dry in some areas', 'Normal, no issues', 'Gets oily quickly']),
    ("How do your pores look?", ['Large and visible, especially on nose', 'Visible only on T-zone', 'Small and barely visible', 'Very small, skin looks tight']),
    ("Reaction to new products?", ['Often gets irritated or red', 'Sometimes breaks out', 'Rarely reacts', 'Absorbs products quickly, needs more']),
    ("Overall skin texture?", ['Rough, flaky or tight', 'Smooth in some areas, oily in others', 'Smooth and balanced overall', 'Consistently shiny and greasy']),
    ("Tissue test result?", ['A lot of oil all over', 'Oil mainly from T-zone', 'Very little oil', 'Almost nothing, skin is dry'])
]

user_inputs = []
for i, (q, opts) in enumerate(QUESTIONS):
    choice = st.selectbox(f"Q{i+1}: {q}", options=[None] + opts, index=0)
    user_inputs.append(choice)

# ── 6. ANALYSIS EXECUTION ──────────────────────────────────────────
if st.button('🚀 Run AI Analysis'):
    if None in user_inputs:
        st.warning("Please answer all questions to allow the ML model to process your data.")
    else:
        with st.spinner('Model classifying and Gemini generating report...'):
            # Prepare data for ML
            encoded_inputs = np.array([VAL_MAP[ans] for ans in user_inputs]).reshape(1, -1)
            
            # Predict
            model = get_trained_model()
            prediction = model.predict(encoded_inputs)[0]
            probs = model.predict_proba(encoded_inputs)
            confidence = round(np.max(probs) * 100, 2)
            
            # Generate Gemini Advice
            ai_report = generate_ai_report(prediction, confidence, user_inputs)
            
            # Display Results
            st.success(f"Analysis Complete: {prediction} Skin identified.")
            
            col1, col2 = st.columns(2)
            col1.metric("Predicted Type", prediction)
            col2.metric("ML Confidence", f"{confidence}%")
            
            st.markdown(f'<div class="report-card"><h3>🩺 AI Dermatologist Consultation</h3>{ai_report}</div>', unsafe_allow_html=True)
            
            st.balloons()

st.divider()
st.caption("Academic Project by [Thinzar Su Hlaing] | Data Science & AI Portfolio | 2026")
