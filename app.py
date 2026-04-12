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

# ── SCORING ─────────────────────────────────────────────────────
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

    return skin_type, percentages

# ── PRODUCTS ─────────────────────────────────────────────────────
PRODUCTS = {
    "Dry": [
        {"name": "CeraVe Moisturizing Cream", "img": "https://i.imgur.com/1bX5QH6.jpg"},
        {"name": "Hyaluronic Acid Serum", "img": "https://i.imgur.com/6YVZ5pM.jpg"},
    ],
    "Oily": [
        {"name": "Niacinamide Serum", "img": "https://i.imgur.com/W2z7K5H.jpg"},
        {"name": "Salicylic Cleanser", "img": "https://i.imgur.com/9R9QZ7T.jpg"},
    ],
    "Normal": [
        {"name": "Neutrogena Hydro Boost", "img": "https://i.imgur.com/Nm3Q2kF.jpg"},
        {"name": "Gentle Cleanser", "img": "https://i.imgur.com/7F2xZ5M.jpg"},
    ],
    "Combination": [
        {"name": "COSRX Cleanser", "img": "https://i.imgur.com/Vc3h1F8.jpg"},
        {"name": "Laneige Cream", "img": "https://i.imgur.com/2x9F6Yc.jpg"},
    ]
}

# ── UI ───────────────────────────────────────────────────────────
st.title('🧴 Skin Type Analyzer AI')
st.markdown('*Answer questions + upload image for smarter result*')

# IMAGE
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
    {
        'q': 'By midday, how does your skin look?',
        'opts': [
            'Very shiny all over',
            'Shiny only on forehead, nose, chin (T-zone)',
            'Still looks the same as morning',
            'Feels drier and tighter',
        ]
    },
    {
        'q': 'How often do you get breakouts?',
        'opts': [
            'Frequently, all over face',
            'Occasionally, mainly T-zone',
            'Rarely',
            'Almost never, but skin feels flaky',
        ]
    },
    {
        'q': 'How does your skin feel without moisturizer?',
        'opts': [
            'Very dry, flaky or itchy',
            'Slightly dry in some areas',
            'Normal, no issues',
            'Gets oily quickly',
        ]
    },
    {
        'q': 'How are your pores?',
        'opts': [
            'Large and visible, especially on nose',
            'Visible only on T-zone',
            'Small and barely visible',
            'Very small, skin looks tight',
        ]
    },
    {
        'q': 'How does your skin react to products?',
        'opts': [
            'Often gets irritated or red',
            'Sometimes breaks out',
            'Rarely reacts',
            'Absorbs products quickly, needs more',
        ]
    },
    {
        'q': 'How is your skin texture?',
        'opts': [
            'Rough, flaky or tight',
            'Smooth in some areas, oily in others',
            'Smooth and balanced overall',
            'Consistently shiny and greasy',
        ]
    },
    {
        'q': 'How much oil appears on blotting paper?',
        'opts': [
            'A lot of oil all over',
            'Oil mainly from T-zone',
            'Very little oil',
            'Almost nothing, skin is dry',
        ]
    }
]

answers = []
all_answered = True

for i, q in enumerate(QUESTIONS):
    ans = st.radio(q['q'], q['opts'], index=None, key=i)
    answers.append(ans)
    if ans is None:
        all_answered = False

# BUTTON + RESULT
if st.button("🔍 Analyze", disabled=not all_answered):

    skin_type, percentages = classify_skin(answers)

    # AI Fusion
    if image_result:
        for k in percentages:
            if k == image_result:
                percentages[k] += 10
            else:
                percentages[k] = max(0, percentages[k] - 2)

        skin_type = max(percentages, key=percentages.get)

    # RESULT UI
    st.markdown(f"""
    <div class="result-box">
        <h2>✨ {skin_type} Skin</h2>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"Confidence: {percentages[skin_type]}%")
    st.progress(percentages[skin_type] / 100)

    for k, v in percentages.items():
        st.write(f"{k}: {v}%")

    st.subheader("📊 Skin Score Chart")
    st.bar_chart(percentages)

    st.subheader("🛍️ Recommended Products")

    cols = st.columns(2)
    for i, product in enumerate(PRODUCTS[skin_type]):
        with cols[i % 2]:
            st.image(product["img"], use_container_width=True)
            st.caption(product["name"])
