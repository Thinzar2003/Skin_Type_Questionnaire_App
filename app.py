import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from google import genai
from huggingface_hub import InferenceClient
import joblib
import os
import matplotlib.pyplot as plt

# ── 1. CONFIG & PROFESSIONAL UI THEME ─────────────────────────────
st.set_page_config(page_title='SkinAI Precision Pro: Research Edition', page_icon='🔬', layout='centered')

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+JP:wght@400;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', 'Noto+Sans+JP', sans-serif; }
    
    .main { background-color: #0e1117; color: #ffffff; }
    
    /* Glassmorphism Card */
    .report-card { 
        background: rgba(255, 255, 255, 0.05); 
        padding: 30px; 
        border-radius: 24px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 25px;
        color: #f0f2f6;
    }
    
    /* Custom Button Gradient */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); 
        color: white; border: none; font-weight: 700; font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ── 2. DATA DICTIONARIES ──────────────────────────────────────────
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

LOCAL_REPORTS = {
    "Normal": {
        "EN": "✨ **Routine:** Cleanse ➔ Hydrate ➔ Protect.\n🌟 **Products:** Hyaluronic Acid, Vitamin C, SPF 50+.",
        "TH": "✨ **ขั้นตอน:** ล้างหน้า ➔ เติมความชุ่มชื้น ➔ ปกป้องผิว\n🌟 **ผลิตภัณฑ์:** ไฮยาลูรอนิก, วิตามินซี, กันแดด SPF 50+",
        "JP": "✨ **お手入れ:** 洗顔 ➔ 保湿 ➔ 紫外線対策\n🌟 **おすすめ:** ヒアルロン酸, ビタミンC, 日焼け止め"
    },
    "Oily": {
        "EN": "🧼 **Routine:** Double Cleanse ➔ Exfoliate (BHA) ➔ Oil-free Gel.\n🌿 **Products:** Niacinamide, Salicylic Acid, Clay Masks.",
        "TH": "🧼 **ขั้นตอน:** ดับเบิลคลีนซิ่ง ➔ ผลัดเซลล์ผิว (BHA) ➔ เจลคุมมัน\n🌿 **ผลิตภัณฑ์:** ไนอะซินาไมด์, ซาลิไซลิก แอซิด, โคลนพอกผิว",
        "JP": "🧼 **お手入れ:** ダブル洗顔 ➔ 角質ケア (BHA) ➔ オイルフリージェル\n🌿 **おすすめ:** ナイアシンアミド, サリチル酸, クレイマスク"
    },
    "Dry": {
        "EN": "🌊 **Routine:** Milky Cleanser ➔ Rich Cream ➔ Face Oil.\n🍯 **Products:** Ceramides, Glycerin, Squalane Oil.",
        "TH": "🌊 **ขั้นตอน:** คลีนเซอร์สูตรน้ำนม ➔ ครีมบำรุงเข้มข้น ➔ ออยล์ทาหน้า\n🍯 **ผลิตภัณฑ์:** เซราไมด์, กลีเซอรีน, สควาเลน ออยล์",
        "JP": "🌊 **お手入れ:** ミルククレンジング ➔ 高保湿クリーム ➔ フェイスオイル\n🍯 **おすすめ:** セラミド, グリセリン, スクワランオイル"
    },
    "Combination": {
        "EN": "⚖️ **Routine:** Balancing Cleanser ➔ Multi-moisturizing ➔ SPF.\n🌸 **Products:** Rosewater, Light Lotions, Targeted Spot Treatment.",
        "TH": "⚖️ **ขั้นตอน:** คลีนเซอร์ปรับสมดุล ➔ บำรุงแยกส่วน ➔ กันแดด\n🌸 **ผลิตภัณฑ์:** น้ำกุหลาบ, โลชั่นเนื้อบางเบา, การแต้มสิวเฉพาะจุด",
        "JP": "⚖️ **お手入れ:** バランスクレンジング ➔ 部位別保湿 ➔ 紫外線対策\n🌸 **おすすめ:** ローズウォーター, 乳液, スポットケア"
    }
}

# ── 3. MACHINE LEARNING ENGINE (Research-Oriented) ──────────────────
def get_trained_model():
    model_filename = 'skin_model.pkl'
    feature_names = ["Wash Feel", "Midday", "Breakout", "No Moist.", "Pores", "Reaction", "Texture", "Tissue"]
    
    if os.path.exists(model_filename):
        return joblib.load(model_filename), feature_names
    
    data = []
    labels = ['Dry', 'Normal', 'Oily', 'Combination']
    for _ in range(1000): # Increased size for better research reliability
        s_type = np.random.choice(labels)
        if s_type == 'Oily': row = [np.random.randint(2,4) for _ in range(8)]
        elif s_type == 'Dry': row = [np.random.randint(0,2) for _ in range(8)]
        else: row = [np.random.randint(0,4) for _ in range(8)]
        data.append(row + [s_type])
    
    df = pd.DataFrame(data, columns=feature_names + ['target'])
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_filename)
    return model, feature_names

# ── 4. RESEARCH ANALYTICS VISUALIZER ──────────────────────────────
def plot_research_metrics(model, feature_names, input_probs, classes):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Importance (RF Weight)**")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=True)
        st.bar_chart(feat_df.set_index('Feature'))

    with col2:
        st.write("**Classification Probability**")
        prob_df = pd.DataFrame({'Type': classes, 'Probability': input_probs[0]})
        st.bar_chart(prob_df.set_index('Type'))

# ── 5. MULTI-MODEL AI LOGIC ───────────────────────────────────────
def generate_ai_report(skin_type, confidence, answers):
    prompt = f"""
    Act as a professional Dermatologist. 
    Analysis: {skin_type} skin ({confidence}% ML confidence).
    User Data: {answers}.
    Provide a professional explanation, a 3-step routine with emojis, and summaries in English, Japanese, and Thai.
    """
    
    # Try Gemini
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return response.text
    except:
        # Try HF Failover
        try:
            hf_client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
            hf_res = hf_client.chat.completions.create(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            return hf_res.choices[0].message.content
        except:
            # Local Failover
            d = LOCAL_REPORTS[skin_type]
            return f"### 🇬🇧 English Analysis\n{d['EN']}\n\n### 🇯🇵 日本語アドバイス\n{d['JP']}\n\n### 🇹🇭 คำแนะนำภาษาไทย\n{d['TH']}"

# ── 6. MAIN UI EXECUTION ──────────────────────────────────────────
st.title('🧴 SkinAI Precision Pro')
st.caption("Hybrid ML & Generative AI for Advanced Dermatological Diagnostics")

user_inputs = []
for i, (q, opts) in enumerate(QUESTIONS):
    st.markdown(f"**{i+1}. {q}**")
    choice = st.selectbox("", options=[None] + opts, key=f"q_{i}", label_visibility="collapsed")
    user_inputs.append(choice)
    st.write("")

if st.button('🚀 RUN RESEARCH-GRADE ANALYSIS'):
    if None in user_inputs:
        st.warning("Please answer all diagnostic questions.")
    else:
        with st.spinner('Calculating Biometric Probabilities...'):
            try:
                # 1. ML Logic
                encoded = np.array([VAL_MAP[ans] for ans in user_inputs]).reshape(1, -1)
                clf, feat_names = get_trained_model()
                prediction = clf.predict(encoded)[0]
                probs = clf.predict_proba(encoded)
                confidence = round(np.max(probs) * 100, 1)
                
                # 2. UI Display
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("📊 Diagnostic Summary")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Type", prediction)
                c2.metric("ML Confidence", f"{confidence}%")
                
                # 3. Research Visuals
                st.markdown("---")
                st.subheader("🔬 Model Interpretability & Research Data")
                plot_research_metrics(clf, feat_names, probs, clf.classes_)
                
                # 4. Expert Report
                st.markdown("---")
                st.subheader("📋 Professional Consultation")
                report = generate_ai_report(prediction, confidence, user_inputs)
                st.write(report)
                st.markdown('</div>', unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"Critical System Failure: {e}")

st.markdown("<br><hr><center>Developed by Thinzar Su Hlaing | Faculty of Digital Innovation Technology | Tokyo Portfolio 2026</center>", unsafe_allow_html=True)
