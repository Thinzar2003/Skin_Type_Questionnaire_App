import streamlit as st
import numpy as np
from PIL import Image
import os

# Optional TensorFlow import (safe)
try:
    from tensorflow.keras.models import load_model
except:
    load_model = None

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title='Skin Type Analyzer',
    page_icon='🧴',
    layout='centered'
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans&display=swap');

html, body {
    font-family: 'DM Sans', sans-serif;
}
h1, h2 {
    font-family: 'DM Serif Display', serif;
}
.main { background-color: #fdf8f4; }

.stButton > button {
    background: #2d2d2d;
    color: white;
    border-radius: 8px;
    width: 100%;
}

.result-box {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    border-left: 5px solid #2d2d2d;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
</style>
''', unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    if load_model is None:
        return None

    if os.path.exists("skin_model.h5"):
        return load_model("skin_model.h5")
    return None

model = load_cnn()

# ── AI IMAGE ANALYSIS ─────────────────────────────────────────────
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

# ── QUIZ SCORING ─────────────────────────────────────────────────
def classify_skin(answers):
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    maps = [
        {
            'Very tight and uncomfortable': {'Dry': 3},
            'Slightly tight': {'Dry': 2, 'Normal': 1},
            'Comfortable and balanced': {'Normal': 3},
            'Fine, no particular feeling': {'Normal': 2, 'Oily': 1},
        },
        {
            'Very shiny all over': {'Oily': 3},
            'Shiny only on forehead, nose, chin (T-zone)': {'Combination': 3},
            'Still looks the same as morning': {'Normal': 3},
            'Feels drier and tighter': {'Dry': 3},
        },
        {
            'Frequently, all over face': {'Oily': 2},
            'Occasionally, mainly T-zone': {'Combination': 2},
            'Rarely': {'Normal': 2},
            'Almost never, but skin feels flaky': {'Dry': 2},
        },
        {
            'Very dry, flaky or itchy': {'Dry': 3},
            'Slightly dry in some areas': {'Combination': 2, 'Dry': 1},
            'Normal, no issues': {'Normal': 3},
            'Gets oily quickly': {'Oily': 3},
        },
        {
            'Large and visible, especially on nose': {'Oily': 2, 'Combination': 1},
            'Visible only on T-zone': {'Combination': 3},
            'Small and barely visible': {'Normal': 2, 'Dry': 1},
            'Very small, skin looks tight': {'Dry': 2},
        },
        {
            'Often gets irritated or red': {'Dry': 2},
            'Sometimes breaks out': {'Oily': 1, 'Combination': 1},
            'Rarely reacts': {'Normal': 2},
            'Absorbs products quickly, needs more': {'Oily': 2},
        },
        {
            'Rough, flaky or tight': {'Dry': 3},
            'Smooth in some areas, oily in others': {'Combination': 3},
            'Smooth and balanced overall': {'Normal': 3},
            'Consistently shiny and greasy': {'Oily': 3},
        },
        {
            'A lot of oil all over': {'Oily': 3},
            'Oil mainly from T-zone': {'Combination': 3},
            'Very little oil': {'Normal': 2},
            'Almost nothing, skin is dry': {'Dry': 3},
        }
    ]

    for i, answer in enumerate(answers):
        for k, v in maps[i].get(answer, {}).items():
            scores[k] += v

    total = sum(scores.values())
    percentages = {k: round(v/total*100, 1) for k, v in scores.items()}
    skin_type = max(scores, key=scores.get)
    confidence = round(scores[skin_type]/total*100, 1)

    return skin_type, percentages, confidence

# ── PRODUCTS ─────────────────────────────────────────────────────
PRODUCTS = {
    "Dry": [
        "CeraVe Moisturizing Cream",
        "La Roche-Posay Hydrating Cleanser",
        "The Ordinary Hyaluronic Acid"
    ],
    "Oily": [
        "The Ordinary Niacinamide 10%",
        "COSRX Salicylic Cleanser",
        "Biore UV Aqua Rich SPF"
    ],
    "Normal": [
        "Neutrogena Hydro Boost",
        "Cetaphil Gentle Cleanser",
        "Innisfree Green Tea Serum"
    ],
    "Combination": [
        "Laneige Water Bank Cream",
        "COSRX Low pH Cleanser",
        "Some By Mi Toner"
    ]
}

# ── QUESTIONS ────────────────────────────────────────────────────
QUESTIONS = [
    ("How does your skin feel after washing?", [
        'Very tight and uncomfortable',
        'Slightly tight',
        'Comfortable and balanced',
        'Fine, no particular feeling'
    ]),
    ("By midday, how does your skin look?", [
        'Very shiny all over',
        'Shiny only on forehead, nose, chin (T-zone)',
        'Still looks the same as morning',
        'Feels drier and tighter'
    ]),
    ("Breakouts?", [
        'Frequently, all over face',
        'Occasionally, mainly T-zone',
        'Rarely',
        'Almost never, but skin feels flaky'
    ]),
    ("Without moisturizer?", [
        'Very dry, flaky or itchy',
        'Slightly dry in some areas',
        'Normal, no issues',
        'Gets oily quickly'
    ]),
    ("Pores?", [
        'Large and visible, especially on nose',
        'Visible only on T-zone',
        'Small and barely visible',
        'Very small, skin looks tight'
    ]),
    ("Reaction to products?", [
        'Often gets irritated or red',
        'Sometimes breaks out',
        'Rarely reacts',
        'Absorbs products quickly, needs more'
    ]),
    ("Texture?", [
        'Rough, flaky or tight',
        'Smooth in some areas, oily in others',
        'Smooth and balanced overall',
        'Consistently shiny and greasy'
    ]),
    ("Blotting paper?", [
        'A lot of oil all over',
        'Oil mainly from T-zone',
        'Very little oil',
        'Almost nothing, skin is dry'
    ])
]

# ── UI ───────────────────────────────────────────────────────────
st.title("🧴 Skin Type Analyzer")
st.markdown("✨ AI + Quiz combined analysis")

# IMAGE
st.subheader("📸 Upload Face Image")
img_file = st.file_uploader("Upload", type=["jpg", "png"])

image_result = None
image_conf = None

if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)

    if model:
        with st.spinner("Analyzing with AI..."):
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
analyze = st.button("🔍 Analyze", disabled=not all_answered)

# RESULT
if analyze:
    quiz_type, percentages, confidence = classify_skin(answers)

    # FINAL DECISION
    final_type = quiz_type
    if image_result:
        final_type = image_result

    # RESULT BOX
    st.markdown(f"""
    <div class="result-box">
        <h2>✨ {final_type} Skin</h2>
        <p>Confidence: {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)

    # AI vs Quiz
    st.markdown("### 🤖 Analysis Breakdown")
    st.write(f"Quiz Result: {quiz_type} ({confidence}%)")

    if image_result:
        st.write(f"AI Result: {image_result} ({image_conf}%)")
        st.write(f"Final Decision: {final_type}")
    else:
        st.write("AI model not available")

    # PRODUCTS
    st.markdown("### 🛍️ Recommended Products")
    for p in PRODUCTS[final_type]:
        st.markdown(f"- {p}")
