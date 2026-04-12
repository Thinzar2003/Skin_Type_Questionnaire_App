import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

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
    border-left: 5px solid;
}
</style>
''', unsafe_allow_html=True)

# ── LOAD AI MODEL ─────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    try:
        return load_model("skin_model.h5")
    except:
        return None

model = load_cnn()

# ── AI IMAGE ANALYSIS ─────────────────────────────────────────────
def analyze_image(img):
    if model is None:
        return None

    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    classes = ["Dry", "Normal", "Oily", "Combination"]

    return classes[np.argmax(pred)]

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
    "Dry": ["CeraVe Moisturizing Cream", "Cetaphil Cleanser", "Hyaluronic Acid Serum"],
    "Oily": ["La Roche-Posay Effaclar", "Niacinamide Serum", "Biore UV Sunscreen"],
    "Normal": ["Neutrogena Hydro Boost", "Simple Face Wash", "Anessa Sunscreen"],
    "Combination": ["COSRX Cleanser", "Some By Mi Toner", "Laneige Cream"]
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
        'Absorbs products quickly'
    ]),
    ("Texture?", [
        'Rough, flaky or tight',
        'Combination areas',
        'Smooth and balanced',
        'Oily and shiny'
    ]),
    ("Blotting paper?", [
        'A lot of oil',
        'T-zone only',
        'Very little oil',
        'Dry'
    ])
]

# ── UI ───────────────────────────────────────────────────────────
st.title("🧴 Skin Type Analyzer")
st.markdown("AI + Quiz combined analysis")

# IMAGE
st.subheader("📸 Upload Image")
img_file = st.file_uploader("Upload", type=["jpg", "png"])

image_result = None
if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)

    if model:
        with st.spinner("AI analyzing..."):
            image_result = analyze_image(img)

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

    final_type = quiz_type
    if image_result:
        final_type = image_result

    st.markdown(f"## Result: {final_type}")
    st.markdown(f"**Confidence: {confidence}%**")

    st.markdown("### 🤖 AI Decision")
    if image_result:
        st.write(f"Quiz: {quiz_type}")
        st.write(f"Image AI: {image_result}")
        st.write(f"Final: {final_type}")
    else:
        st.write("Quiz only result")

    st.markdown("### 🛍️ Recommended Products")
    for p in PRODUCTS[final_type]:
        st.markdown(f"- {p}")
