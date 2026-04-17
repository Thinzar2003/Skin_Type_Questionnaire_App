import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from google import genai
import joblib
import os

# ── 1. CONFIG & SECURITY ──────────────────────────────────────────
st.set_page_config(page_title='AI Skin Analyzer Pro', page_icon='🧴', layout='centered')

api_key = st.secrets.get("GEMINI_API_KEY")

# ── 2. AI MODEL LOGIC (PREDICTIVE) ────────────────────────────────
def get_trained_model():
    model_filename = 'skin_model.pkl'
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    
    # Synthetic dataset for Research Validation
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
    joblib.dump(model, model_filename)
    return model

# ── 3. GENERATIVE AI LOGIC (NEW SDK 2026) ──────────────────────────
def generate_ai_report(skin_type, confidence, answers):
    try:
        # Initializing the new 2026 Client
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Act as a professional Dermatologist for an academic project. 
        Analysis: {skin_type} skin ({confidence}% confidence).
        Data: {answers}.
        
        Provide a biological explanation, 3-step routine, and a brief Thai summary.
        """

        # Using the updated model name strings
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )
        return response.text
            
    except Exception as e:
        return f"Model Connection Error: {str(e)}"

# ── 4. UI STYLING ──────────────────────────────────────────────────
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stButton>button { background-color: #2d2d2d; color: white; border-radius: 10px; height: 3.5em; font-weight: 700; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-top: 5px solid #2d2d2d; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-top: 20px; color: #333; }
</style>
''', unsafe_allow_html=True)

# ── 5. APPLICATION UI ─────────────────────────────────────────────
st.title('🧴 AI Skin Type Research Pro')
st.info('**Methodology:** This Hybrid AI system combines a *Random Forest Classifier* with *Gemini 2.0*.')

# ... (Insert your QUESTIONS and VAL_MAP from previous versions here)

user_inputs = []
for i, (q, opts) in enumerate(QUESTIONS):
    choice = st.selectbox(f"Q{i+1}: {q}", options=[None] + opts, index=0)
    user_inputs.append(choice)

if st.button('🚀 Execute Hybrid AI Analysis'):
    if None in user_inputs:
        st.warning("Please answer all questions.")
    elif not api_key:
        st.error("API Key not found in Streamlit Secrets.")
    else:
        with st.spinner('Running Inference...'):
            encoded_inputs = np.array([VAL_MAP[ans] for ans in user_inputs]).reshape(1, -1)
            clf = get_trained_model()
            prediction = clf.predict(encoded_inputs)[0]
            confidence = round(np.max(clf.predict_proba(encoded_inputs)) * 100, 2)
            ai_report = generate_ai_report(prediction, confidence, user_inputs)
            
            st.success("Classification Complete")
            c1, c2 = st.columns(2)
            c1.metric("Clinical Type", prediction)
            c2.metric("ML Accuracy Index", f"{confidence}%")
            st.markdown(f'<div class="report-card"><h3>📋 Expert System Report</h3>{ai_report}</div>', unsafe_allow_html=True)
            st.balloons()
