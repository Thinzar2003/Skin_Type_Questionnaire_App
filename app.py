import streamlit as st
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ── Page config ─────────────────────────────
st.set_page_config(page_title="Skin AI Analyzer", page_icon="🧴")

# ── CSS ────────────────────────────────────
st.markdown("""
<style>
body {background:#fdf8f4;}
.result {
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("🧴 Skin Type AI Analyzer")

# ── Fake Training Data (for demo AI) ───────
X = [
    [200,50],[180,40],[100,30],[90,20],
    [220,60],[150,35],[120,25],[80,15]
]
y = ["Oily","Oily","Normal","Dry","Oily","Normal","Combination","Dry"]

model = RandomForestClassifier()
model.fit(X,y)

# ── Image Feature Extract ─────────────────
def extract_features(img):
    arr = np.array(img)
    brightness = np.mean(arr)
    contrast = np.std(arr)
    return [brightness, contrast]

# ── UI Tabs ───────────────────────────────
tab1, tab2 = st.tabs(["📝 Questionnaire","📸 AI Image"])

# ── TAB 1 ────────────────────────────────
with tab1:
    st.subheader("Answer Questions")

    q1 = st.selectbox("Skin feel?", ["Dry","Normal","Oily","Combination"])
    q2 = st.selectbox("Midday look?", ["Dry","Normal","Oily","Combination"])

    if st.button("Analyze Questionnaire"):
        st.success(f"Result: {q1}")

# ── TAB 2 ────────────────────────────────
with tab2:
    st.subheader("Upload Image")

    file = st.file_uploader("Upload face image")

    if file:
        img = Image.open(file)
        st.image(img, use_column_width=True)

        features = extract_features(img)

        if st.button("Run AI Analysis"):
            pred = model.predict([features])[0]

            st.markdown(f"""
            <div class="result">
            <h2>✨ AI Result: {pred}</h2>
            <p>Brightness: {round(features[0],1)}</p>
            <p>Contrast: {round(features[1],1)}</p>
            </div>
            """, unsafe_allow_html=True)
