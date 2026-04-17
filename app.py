import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import joblib
import os

# ── 1. CONFIG & SECURITY ──────────────────────────────────────────
st.set_page_config(page_title='AI Skin Analyzer Pro', page_icon='🧴', layout='centered')

# Access the API Key securely from Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing Gemini API Key. Please add it to Streamlit Secrets.")

# ── 2. AI MODEL LOGIC (PREDICTIVE) ────────────────────────────────
def get_trained_model():
    model_filename = 'skin_model.pkl'
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    
    # Synthetic dataset generation for Academic Validation
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

# ── 3. GENERATIVE AI LOGIC (CONSULTATION) ──────────────────────────
def generate_ai_report(skin_type, confidence, answers):
    try:
        # 1. Ensure the key is loaded
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not found in Streamlit Secrets."
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 2. Refined Academic Prompt
        prompt = f"""
        Act as a professional Dermatologist. 
        Classification Result: {skin_type} skin.
        Model Confidence: {confidence}%.
        
        Provide:
        1. Biological explanation for this classification.
        2. A 3-step skincare protocol.
        3. A summary in Thai for the user.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # This will print the specific error (like 401) to your UI for debugging
        return f"AI engine error: {str(e)}"

# ── 4. UI STYLING ──────────────────────────────────────────────────
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stButton>button { background-color: #2d2d2d; color: white; border-radius: 10px; height: 3.5em; width: 100%; font-weight: 700; }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-top: 5px solid #2d2d2d; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-top: 20px; color: #333; }
</style>
''', unsafe_allow_html=True)

# ── 5. APPLICATION UI ─────────────────────────────────────────────
st.title('🧴 AI Skin Type Research Pro')
st.info('**Academic Methodology:** This project demonstrates a Hybrid AI system combining a *Random Forest Classifier* with *Gemini 1.5*.')

VAL_MAP = {
    'Very tight and uncomfortable': 0, 'Slightly tight': 1, 'Comfortable and balanced': 2, 'Fine, no particular feeling': 3,
    'Very shiny all over': 3, 'Shiny only on forehead, nose, chin (T-zone)': 2, 'Still looks the same as morning': 1, 'Feels drier and tighter': 0,
    'Frequently, all over face': 3, 'Occasionally, mainly T-zone': 2, 'Rarely': 1, 'Almost never, but skin feels flaky': 0,
    'Very dry, flaky or itchy': 0, 'Slightly dry in some areas': 1, 'Normal, no issues': 2, 'Gets oily quickly': 3,
    'Large and visible, especially on nose': 3, 'Visible only on T-zone': 2, 'Small and barely visible': 1, 'Very small, skin looks tight': 0,
    'Often gets irritated or red': 0, 'Sometimes breaks out': 2, 'Rarely reacts': 1, 'Absorbs products quickly, needs more': 3,
    'Rough, flaky or tight': 0, 'Smooth in some areas, oily in others': 2, 'Smooth and balanced overall': 1, 'Consistently shiny and greasy': 3,
    'A lot of oil all over': 3, 'Oil mainly from T-zone': 2, 'Very little oil': 1, 'Almost nothing, skin is dry': 0
}

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

if st.button('🚀 Execute Hybrid AI Analysis'):
    if None in user_inputs:
        st.warning("All features must be populated for model inference.")
    else:
        with st.spinner('Calculating Model Probabilities...'):
            encoded_inputs = np.array([VAL_MAP[ans] for ans in user_inputs]).reshape(1, -1)
            clf = get_trained_model()
            prediction = clf.predict(encoded_inputs)[0]
            confidence = round(np.max(clf.predict_proba(encoded_inputs)) * 100, 2)
            ai_report = generate_ai_report(prediction, confidence, user_inputs)
            
            st.success(f"Classification Successful")
            c1, c2 = st.columns(2)
            c1.metric("Clinical Type", prediction)
            c2.metric("ML Accuracy Index", f"{confidence}%")
            st.markdown(f'<div class="report-card"><h3>📋 Expert System Report</h3>{ai_report}</div>', unsafe_allow_html=True)
            st.balloons()

st.divider()
st.caption("Developed by Thinzar Su Hlaing | Faculty of Data Science | Academic Research 2026")
