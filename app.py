import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ─────────────────────────────
st.set_page_config(page_title="Skin Type Analyzer", page_icon="🧴", layout="centered")

# ── CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #fdf8f4;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
.result-box {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-top: 10px;
}
.score-bar {
    height: 8px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Skin scoring ────────────────────────────
def classify_skin(answers):
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    mapping = [
        {'Dry':3},{'Oily':3},{'Normal':3},{'Combination':3}
    ]

    for ans in answers:
        scores[ans] += 2

    total = sum(scores.values()) or 1
    pct = {k: round(v/total*100,1) for k,v in scores.items()}
    skin = max(scores, key=scores.get)
    conf = round(scores[skin]/total*100,1)

    return skin, pct, conf

# ── AI-like image analysis ──────────────────
def analyze_image(img):
    arr = np.array(img)

    brightness = np.mean(arr)
    contrast = np.std(arr)

    scores = {'Dry':0,'Normal':0,'Oily':0,'Combination':0}

    if brightness > 180:
        scores['Oily'] += 3
    elif brightness > 130:
        scores['Normal'] += 2
    else:
        scores['Dry'] += 2

    if contrast > 50:
        scores['Combination'] += 2
    else:
        scores['Normal'] += 1

    total = sum(scores.values()) or 1
    pct = {k: round(v/total*100,1) for k,v in scores.items()}
    skin = max(scores, key=scores.get)
    conf = round(scores[skin]/total*100,1)

    features = {
        "Brightness": round(brightness,1),
        "Contrast": round(contrast,1)
    }

    return skin, pct, conf, features

# ── UI ──────────────────────────────────────
st.title("🧴 Skin Type Analyzer")
st.markdown("Discover your skin type using questionnaire and AI image analysis")

tab1, tab2 = st.tabs(["📝 Questionnaire", "📸 Image AI"])

# ── TAB 1 ──────────────────────────────────
with tab1:
    st.subheader("Answer questions")

    options = ["Dry","Normal","Oily","Combination"]

    answers = []
    for i in range(5):
        ans = st.selectbox(f"Question {i+1}", options, key=i)
        answers.append(ans)

    if st.button("Analyze Questionnaire"):
        skin, pct, conf = classify_skin(answers)

        st.markdown(f"""
        <div class="result-box">
        <h2>✨ {skin} Skin</h2>
        <p>Confidence: {conf}%</p>
        </div>
        """, unsafe_allow_html=True)

        for k,v in pct.items():
            st.write(f"{k}: {v}%")

# ── TAB 2 ──────────────────────────────────
with tab2:
    st.subheader("Upload your photo")

    file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, use_column_width=True)

        if st.button("Run AI Analysis"):
            skin, pct, conf, features = analyze_image(img)

            st.markdown(f"""
            <div class="result-box">
            <h2>🤖 AI Result: {skin}</h2>
            <p>Confidence: {conf}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### Image Features")
            st.write(features)

            st.write("### Score")
            for k,v in pct.items():
                st.write(f"{k}: {v}%")
