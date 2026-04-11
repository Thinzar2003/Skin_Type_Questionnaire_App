import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title='Skin Type Analyzer',
    page_icon='🧴',
    layout='centered'
)

# ── Simple CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    padding: 1.5rem;
    border-radius: 12px;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────
COLORS = {
    'Dry': '#3b82f6',
    'Normal': '#10b981',
    'Oily': '#f59e0b',
    'Combination': '#8b5cf6'
}

# ── Questionnaire logic ───────────────────────────────────────────────
def classify_by_questionnaire(answers):
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    for ans in answers:
        if ans == "Dry":
            scores["Dry"] += 2
        elif ans == "Oily":
            scores["Oily"] += 2
        elif ans == "Normal":
            scores["Normal"] += 2
        else:
            scores["Combination"] += 2

    total = sum(scores.values()) or 1
    best = max(scores, key=scores.get)
    conf = round(scores[best] / total * 100, 1)

    return best, scores, conf


# ── Image analysis (NO OpenCV) ─────────────────────────────────────────
def analyze_image(pil_img):
    img = np.array(pil_img.convert('RGB'))

    brightness = float(np.mean(img))
    contrast = float(np.std(img))

    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    if brightness > 170:
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
    best = max(scores, key=scores.get)
    conf = round(scores[best] / total * 100, 1)

    features = {
        "Brightness": round(brightness, 1),
        "Contrast": round(contrast, 1)
    }

    return best, scores, conf, features


# ── UI ─────────────────────────────────────────────────────────────────
st.title("🧴 Skin Type Analyzer")

tab1, tab2 = st.tabs(["📝 Questionnaire", "📸 Image"])

# ── TAB 1 ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Quick Skin Questions")

    q1 = st.selectbox("How does your skin feel?", ["Dry", "Normal", "Oily", "Combination"])
    q2 = st.selectbox("How does it look midday?", ["Dry", "Normal", "Oily", "Combination"])

    if st.button("Analyze Questionnaire"):
        result, scores, conf = classify_by_questionnaire([q1, q2])
        st.success(f"Result: {result} ({conf}%)")


# ── TAB 2 ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload Image")

    uploaded = st.file_uploader("Upload face image", type=["jpg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

        if st.button("Analyze Image"):
            result, scores, conf, features = analyze_image(img)

            st.success(f"Result: {result} ({conf}%)")
            st.write(features)
