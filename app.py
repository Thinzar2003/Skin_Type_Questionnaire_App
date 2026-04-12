import streamlit as st
import numpy as np
from PIL import Image
import os

# Safe TensorFlow import
try:
    from tensorflow.keras.models import load_model
except:
    load_model = None

# ── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title='Skin Type Analyzer',
    page_icon='🧴',
    layout='centered'
)

# ── CUSTOM CSS ─────────────────────────────────────
st.markdown("""
<style>
body {font-family: sans-serif;}
.main { background-color: #fdf8f4; }

.stButton > button {
    background: #2d2d2d;
    color: white;
    border-radius: 10px;
    width: 100%;
}

.result-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────
@st.cache_resource
def load_cnn():
    if load_model is None:
        return None
    if os.path.exists("skin_model.h5"):
        return load_model("skin_model.h5")
    return None

model = load_cnn()

# ── AI IMAGE ANALYSIS ──────────────────────────────
def analyze_image(img):
    if model is None:
        return None, None

    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    classes = ["Dry", "Normal", "Oily", "Combination"]

    idx = np.argmax(pred)
    confidence = round(float(np.max(pred)) * 100, 1)

    return classes[idx], confidence

# ── QUIZ LOGIC ─────────────────────────────────────
def classify_skin(answers):
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    maps = [
        {'Very tight and uncomfortable': {'Dry': 3},
         'Slightly tight': {'Dry': 2, 'Normal': 1},
         'Comfortable and balanced': {'Normal': 3},
         'Fine, no particular feeling': {'Normal': 2, 'Oily': 1}},

        {'Very shiny all over': {'Oily': 3},
         'Shiny only on forehead, nose, chin (T-zone)': {'Combination': 3},
         'Still looks the same as morning': {'Normal': 3},
         'Feels drier and tighter': {'Dry': 3}},

        {'Frequently, all over face': {'Oily': 2},
         'Occasionally, mainly T-zone': {'Combination': 2},
         'Rarely': {'Normal': 2},
         'Almost never, but skin feels flaky': {'Dry': 2}},

        {'Very dry, flaky or itchy': {'Dry': 3},
         'Slightly dry in some areas': {'Combination': 2, 'Dry': 1},
         'Normal, no issues': {'Normal': 3},
         'Gets oily quickly': {'Oily': 3}},

        {'Large and visible, especially on nose': {'Oily': 2, 'Combination': 1},
         'Visible only on T-zone': {'Combination': 3},
         'Small and barely visible': {'Normal': 2, 'Dry': 1},
         'Very small, skin looks tight': {'Dry': 2}},

        {'Often gets irritated or red': {'Dry': 2},
         'Sometimes breaks out': {'Oily': 1, 'Combination': 1},
         'Rarely reacts': {'Normal': 2},
         'Absorbs products quickly, needs more': {'Oily': 2}},

        {'Rough, flaky or tight': {'Dry': 3},
         'Smooth in some areas, oily in others': {'Combination': 3},
         'Smooth and balanced overall': {'Normal': 3},
         'Consistently shiny and greasy': {'Oily': 3}},

        {'A lot of oil all over': {'Oily': 3},
         'Oil mainly from T-zone': {'Combination': 3},
         'Very little oil': {'Normal': 2},
         'Almost nothing, skin is dry': {'Dry': 3}}
    ]

    for i, answer in enumerate(answers):
        for k, v in maps[i].get(answer, {}).items():
            scores[k] += v

    total = sum(scores.values())
    percentages = {k: round(v / total * 100, 1) for k, v in scores.items()}
    skin_type = max(scores, key=scores.get)
    confidence = round(scores[skin_type] / total * 100, 1)

    return skin_type, percentages, confidence

# ── PRODUCTS ───────────────────────────────────────
PRODUCTS = {
    "Dry": ["CeraVe Moisturizing Cream", "Hyaluronic Acid Serum"],
    "Oily": ["Niacinamide Serum", "Salicylic Cleanser"],
    "Normal": ["Neutrogena Hydro Boost", "Gentle Cleanser"],
    "Combination": ["COSRX Cleanser", "Laneige Cream"]
}

# ── QUESTIONS ──────────────────────────────────────
QUESTIONS = [
    ("After washing?", ['Very tight and uncomfortable','Slightly tight','Comfortable and balanced','Fine']),
    ("Midday skin?", ['Very shiny all over','T-zone only','Same as morning','Feels dry']),
    ("Breakouts?", ['Frequent','Sometimes','Rare','Almost never']),
    ("Without moisturizer?", ['Very dry','Slightly dry','Normal','Gets oily']),
    ("Pores?", ['Large','T-zone','Small','Very small']),
    ("Reaction?", ['Irritated','Breakout','Rare','Absorbs fast']),
    ("Texture?", ['Rough','Mixed','Smooth','Oily']),
    ("Blotting?", ['A lot','T-zone','Little','Dry'])
]

# ── UI ─────────────────────────────────────────────
st.title("🧴 Skin Type Analyzer")
st.markdown("AI + Quiz Analysis")

# IMAGE
img_file = st.file_uploader("Upload Face Image", type=["jpg","png"])

image_result, image_conf = None, None

if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)

    if model:
        with st.spinner("AI analyzing..."):
            image_result, image_conf = analyze_image(img)

# QUIZ
answers = []
all_answered = True

for i, (q, opts) in enumerate(QUESTIONS):
    ans = st.radio(q, opts, key=i, index=None)
    answers.append(ans)
    if ans is None:
        all_answered = False

# BUTTON
if st.button("🔍 Analyze", disabled=not all_answered):

    quiz_type, percentages, confidence = classify_skin(answers)

    # SMART COMBINE
    if image_result:
        if image_conf > 70:
            final_type = image_result
        else:
            final_type = quiz_type
    else:
        final_type = quiz_type

    # RESULT
    st.markdown(f"""
    <div class="result-box">
        <h2>{final_type} Skin</h2>
        <p>Confidence: {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)

    # CHART
    st.subheader("📊 Skin Score")
    st.bar_chart(percentages)

    # BREAKDOWN
    st.subheader("🤖 AI vs Quiz")
    st.write(f"Quiz: {quiz_type} ({confidence}%)")

    if image_result:
        st.write(f"AI: {image_result} ({image_conf}%)")

    # PRODUCTS
    st.subheader("🛍️ Recommended Products")
    for p in PRODUCTS[final_type]:
        st.write(f"- {p}")
