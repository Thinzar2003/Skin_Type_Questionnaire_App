import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title='Skin Type Analyzer AI',
    page_icon='🧴',
    layout='centered'
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans&display=swap');

html, body, [class*='css'] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}
.main { background-color: #fdf8f4; }

.stButton > button {
    background: #2d2d2d;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    width: 100%;
}

.result-box {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border-left: 5px solid;
}

.score-bar {
    height: 8px;
    border-radius: 4px;
    margin: 4px 0 12px 0;
}
</style>
''', unsafe_allow_html=True)

# ── IMAGE AI ─────────────────────────────────────────────────────
def analyze_image(img):
    arr = np.array(img)
    brightness = np.mean(arr)

    if brightness > 180:
        return "Oily"
    elif brightness < 80:
        return "Dry"
    else:
        return "Normal"

# ── ORIGINAL SCORING ─────────────────────────────────────────────
def classify_skin(answers):
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    q1_map = {
        'Very tight and uncomfortable': {'Dry': 3},
        'Slightly tight': {'Dry': 2, 'Normal': 1},
        'Comfortable and balanced': {'Normal': 3},
        'Fine, no particular feeling': {'Normal': 2, 'Oily': 1},
    }
    for k, v in q1_map.get(answers[0], {}).items():
        scores[k] += v

    q2_map = {
        'Very shiny all over': {'Oily': 3},
        'Shiny only on forehead, nose, chin (T-zone)': {'Combination': 3},
        'Still looks the same as morning': {'Normal': 3},
        'Feels drier and tighter': {'Dry': 3},
    }
    for k, v in q2_map.get(answers[1], {}).items():
        scores[k] += v

    q3_map = {
        'Frequently, all over face': {'Oily': 2},
        'Occasionally, mainly T-zone': {'Combination': 2},
        'Rarely': {'Normal': 2},
        'Almost never, but skin feels flaky': {'Dry': 2},
    }
    for k, v in q3_map.get(answers[2], {}).items():
        scores[k] += v

    q4_map = {
        'Very dry, flaky or itchy': {'Dry': 3},
        'Slightly dry in some areas': {'Combination': 2, 'Dry': 1},
        'Normal, no issues': {'Normal': 3},
        'Gets oily quickly': {'Oily': 3},
    }
    for k, v in q4_map.get(answers[3], {}).items():
        scores[k] += v

    q5_map = {
        'Large and visible, especially on nose': {'Oily': 2, 'Combination': 1},
        'Visible only on T-zone': {'Combination': 3},
        'Small and barely visible': {'Normal': 2, 'Dry': 1},
        'Very small, skin looks tight': {'Dry': 2},
    }
    for k, v in q5_map.get(answers[4], {}).items():
        scores[k] += v

    q6_map = {
        'Often gets irritated or red': {'Dry': 2},
        'Sometimes breaks out': {'Oily': 1, 'Combination': 1},
        'Rarely reacts': {'Normal': 2},
        'Absorbs products quickly, needs more': {'Oily': 2},
    }
    for k, v in q6_map.get(answers[5], {}).items():
        scores[k] += v

    q7_map = {
        'Rough, flaky or tight': {'Dry': 3},
        'Smooth in some areas, oily in others': {'Combination': 3},
        'Smooth and balanced overall': {'Normal': 3},
        'Consistently shiny and greasy': {'Oily': 3},
    }
    for k, v in q7_map.get(answers[6], {}).items():
        scores[k] += v

    q8_map = {
        'A lot of oil all over': {'Oily': 3},
        'Oil mainly from T-zone': {'Combination': 3},
        'Very little oil': {'Normal': 2},
        'Almost nothing, skin is dry': {'Dry': 3},
    }
    for k, v in q8_map.get(answers[7], {}).items():
        scores[k] += v

    total = sum(scores.values())
    percentages = {k: round(v/total*100, 1) for k, v in scores.items()}
    skin_type = max(scores, key=scores.get)

    return skin_type, percentages

# ── UI ────────────────────────────────────────────────────────────
st.title('🧴 Skin Type Analyzer AI')
st.markdown('*Answer questions + upload image for smarter result*')

# IMAGE UPLOAD
st.subheader("📸 Upload Image (Optional)")
img_file = st.file_uploader("Upload face image", type=["jpg","png","jpeg"])

image_result = None
if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)
    image_result = analyze_image(img)
    st.success(f"AI Image Prediction: {image_result}")

st.divider()

# QUESTIONS
QUESTIONS = [
    {
        'q': 'How does your skin feel about 1 hour after washing?',
        'opts': [
            'Very tight and uncomfortable',
            'Slightly tight',
            'Comfortable and balanced',
            'Fine, no particular feeling',
        ]
    },
]*8

answers = []
all_answered = True

for i, q in enumerate(QUESTIONS):
    ans = st.radio(q['q'], q['opts'], index=None, key=i)
    answers.append(ans)
    if ans is None:
        all_answered = False

# BUTTON
if st.button("🔍 Analyze", disabled=not all_answered):

    skin_type, percentages = classify_skin(answers)

    # 🔥 AI Fusion
    if image_result:
        percentages[image_result] += 10
        skin_type = max(percentages, key=percentages.get)

    st.subheader(f"✨ Result: {skin_type}")

    for k,v in percentages.items():
        st.write(f"{k}: {v}%")
