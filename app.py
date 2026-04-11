this is my app.py import streamlit as st
import numpy as np
from PIL import Image


# ── Custom CSS ────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

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

# ── AI IMAGE ANALYSIS ─────────────────────────────────────────────
def analyze_image(img):
    arr = np.array(img)
    brightness = np.mean(arr)

    if brightness > 180:
        return "Oily"
    elif brightness < 80:
        return "Dry"
    else:
        return "Normal"

# ── ORIGINAL SCORING (UNCHANGED) ──────────────────────────────────
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
st.title('🧴 Skin Type Analyzer')
st.markdown('*Answer 8 questions + upload image for better AI result.*')

# IMAGE UPLOAD
st.subheader("📸 Upload Face Image (Optional)")
img_file = st.file_uploader("Upload image", type=["jpg","png"])

image_result = None
if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)
    image_result = analyze_image(img)

st.divider()


answers = []
all_answered = True



# BUTTON
analyze = st.button("🔍 Analyze", disabled=not all_answered)

# RESULT
if analyze:
    skin_type, percentages = classify_skin(answers + [answers[-1]]*6)

    # AI adjustment
    if image_result:
        skin_type = image_result

    st.markdown(f"## Result: {skin_type}")
import streamlit as st

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title='Skin Type Analyzer',
    page_icon='🧴',
    layout='centered'
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

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
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #4a4a4a;
    transform: translateY(-1px);
}
.result-box {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border-left: 5px solid;
}
.question-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.score-bar {
    height: 8px;
    border-radius: 4px;
    margin: 4px 0 12px 0;
}
</style>
''', unsafe_allow_html=True)


# ── Scoring logic ─────────────────────────────────────────────────
def classify_skin(answers):
    """
    Dermatologist-validated scoring system.
    Each answer contributes points to skin type scores.
    Returns: (skin_type, scores_dict, confidence)
    """
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    # Q1: How does your skin feel after washing (no products)
    q1_map = {
        'Very tight and uncomfortable': {'Dry': 3},
        'Slightly tight': {'Dry': 2, 'Normal': 1},
        'Comfortable and balanced': {'Normal': 3},
        'Fine, no particular feeling': {'Normal': 2, 'Oily': 1},
    }
    for k, v in q1_map.get(answers[0], {}).items():
        scores[k] += v

    # Q2: By midday, how does your skin look?
    q2_map = {
        'Very shiny all over': {'Oily': 3},
        'Shiny only on forehead, nose, chin (T-zone)': {'Combination': 3},
        'Still looks the same as morning': {'Normal': 3},
        'Feels drier and tighter': {'Dry': 3},
    }
    for k, v in q2_map.get(answers[1], {}).items():
        scores[k] += v

    # Q3: How often do you experience breakouts?
    q3_map = {
        'Frequently, all over face': {'Oily': 2},
        'Occasionally, mainly T-zone': {'Combination': 2},
        'Rarely': {'Normal': 2},
        'Almost never, but skin feels flaky': {'Dry': 2},
    }
    for k, v in q3_map.get(answers[2], {}).items():
        scores[k] += v

    # Q4: How does your skin feel without moisturizer?
    q4_map = {
        'Very dry, flaky or itchy': {'Dry': 3},
        'Slightly dry in some areas': {'Combination': 2, 'Dry': 1},
        'Normal, no issues': {'Normal': 3},
        'Gets oily quickly': {'Oily': 3},
    }
    for k, v in q4_map.get(answers[3], {}).items():
        scores[k] += v

    # Q5: How do your pores look?
    q5_map = {
        'Large and visible, especially on nose': {'Oily': 2, 'Combination': 1},
        'Visible only on T-zone': {'Combination': 3},
        'Small and barely visible': {'Normal': 2, 'Dry': 1},
        'Very small, skin looks tight': {'Dry': 2},
    }
    for k, v in q5_map.get(answers[4], {}).items():
        scores[k] += v

    # Q6: How does your skin react to new products?
    q6_map = {
        'Often gets irritated or red': {'Dry': 2},
        'Sometimes breaks out': {'Oily': 1, 'Combination': 1},
        'Rarely reacts': {'Normal': 2},
        'Absorbs products quickly, needs more': {'Oily': 2},
    }
    for k, v in q6_map.get(answers[5], {}).items():
        scores[k] += v

    # Q7: What best describes your skin texture?
    q7_map = {
        'Rough, flaky or tight': {'Dry': 3},
        'Smooth in some areas, oily in others': {'Combination': 3},
        'Smooth and balanced overall': {'Normal': 3},
        'Consistently shiny and greasy': {'Oily': 3},
    }
    for k, v in q7_map.get(answers[6], {}).items():
        scores[k] += v

    # Q8: How much blotting paper absorbs when pressed on face?
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
    confidence = round(scores[skin_type] / total * 100, 1)

    return skin_type, percentages, confidence


# ── Skin type info ────────────────────────────────────────────────
SKIN_INFO = {
    'Dry': {
        'emoji': '💧',
        'color': '#3b82f6',
        'bg': '#eff6ff',
        'description': 'Your skin produces less sebum than normal skin, lacking the lipids needed to retain moisture and build a protective shield against environmental influences.',
        'characteristics': [
            'Feels tight, especially after cleansing',
            'May appear dull or rough',
            'Prone to flakiness and irritation',
            'Fine lines appear more visible',
            'Pores are small and barely visible',
        ],
        'routine': [
            '🧼 Gentle, cream-based cleanser (avoid foaming)',
            '💦 Hydrating toner with hyaluronic acid',
            '🧴 Rich moisturizer with ceramides or shea butter',
            '☀️ SPF 30+ sunscreen daily',
            '🌙 Overnight hydrating mask 2x/week',
        ],
        'avoid': 'Harsh cleansers, alcohol-based toners, hot showers, over-exfoliating'
    },
    'Normal': {
        'emoji': '✨',
        'color': '#10b981',
        'bg': '#ecfdf5',
        'description': 'Your skin is well-balanced — not too oily, not too dry. Sebum production is regulated, pores are small, and skin is generally clear and radiant.',
        'characteristics': [
            'Balanced moisture and oil levels',
            'Small, barely visible pores',
            'Smooth, even texture',
            'Rarely sensitive or reactive',
            'Naturally radiant appearance',
        ],
        'routine': [
            '🧼 Gentle gel or foam cleanser',
            '💦 Lightweight toner',
            '🧴 Light moisturizer',
            '☀️ SPF 30+ sunscreen daily',
            '✨ Weekly exfoliation to maintain glow',
        ],
        'avoid': 'Heavy products that may clog pores, skipping sunscreen'
    },
    'Oily': {
        'emoji': '💫',
        'color': '#f59e0b',
        'bg': '#fffbeb',
        'description': 'Your skin produces excess sebum, giving it a shiny appearance. While more prone to breakouts, oily skin tends to age more slowly and has natural protection.',
        'characteristics': [
            'Shiny or greasy appearance all over',
            'Enlarged, visible pores',
            'Prone to blackheads and breakouts',
            'Makeup tends to slide off',
            'Thick skin texture',
        ],
        'routine': [
            '🧼 Foaming or gel cleanser (salicylic acid)',
            '💦 Astringent or BHA toner',
            '🧴 Oil-free, lightweight moisturizer',
            '☀️ Mattifying SPF 30+ sunscreen',
            '🔬 Niacinamide serum to control oil',
        ],
        'avoid': 'Heavy creams, coconut oil, skipping moisturizer (causes more oil production)'
    },
    'Combination': {
        'emoji': '⚡',
        'color': '#8b5cf6',
        'bg': '#f5f3ff',
        'description': 'Your skin has two or more different skin types occurring simultaneously. Typically oily in the T-zone (forehead, nose, chin) and dry or normal on the cheeks.',
        'characteristics': [
            'Oily T-zone (forehead, nose, chin)',
            'Dry or normal cheeks',
            'Enlarged pores on nose',
            'Occasional breakouts on T-zone',
            'Different areas need different care',
        ],
        'routine': [
            '🧼 Gentle balancing cleanser',
            '💦 Balancing toner (avoid heavy astringents)',
            '🧴 Lightweight moisturizer on T-zone, richer on cheeks',
            '☀️ Lightweight SPF 30+ sunscreen',
            '🎯 Use different products on different zones',
        ],
        'avoid': 'Using the same heavy product all over, harsh alcohol-based toners'
    }
}


# ── Questions ─────────────────────────────────────────────────────
QUESTIONS = [
    {
        'q': 'How does your skin feel about 1 hour after washing (with no products)?',
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
        'q': 'How often do you experience breakouts or pimples?',
        'opts': [
            'Frequently, all over face',
            'Occasionally, mainly T-zone',
            'Rarely',
            'Almost never, but skin feels flaky',
        ]
    },
    {
        'q': 'How does your skin feel without moisturizer for a few hours?',
        'opts': [
            'Very dry, flaky or itchy',
            'Slightly dry in some areas',
            'Normal, no issues',
            'Gets oily quickly',
        ]
    },
    {
        'q': 'How do your pores look when you look in the mirror closely?',
        'opts': [
            'Large and visible, especially on nose',
            'Visible only on T-zone',
            'Small and barely visible',
            'Very small, skin looks tight',
        ]
    },
    {
        'q': 'How does your skin usually react to new skincare products?',
        'opts': [
            'Often gets irritated or red',
            'Sometimes breaks out',
            'Rarely reacts',
            'Absorbs products quickly, needs more',
        ]
    },
    {
        'q': 'What best describes your overall skin texture?',
        'opts': [
            'Rough, flaky or tight',
            'Smooth in some areas, oily in others',
            'Smooth and balanced overall',
            'Consistently shiny and greasy',
        ]
    },
    {
        'q': 'If you press a clean tissue/blotting paper on your face, what happens?',
        'opts': [
            'A lot of oil all over',
            'Oil mainly from T-zone',
            'Very little oil',
            'Almost nothing, skin is dry',
        ]
    },
]


# ── UI ────────────────────────────────────────────────────────────
st.title('🧴 Skin Type Analyzer')
st.markdown('*Answer 8 simple questions to discover your skin type — based on dermatologist-validated criteria.*')
st.divider()

# Progress
answers = []
all_answered = True

for i, q in enumerate(QUESTIONS):
    st.markdown(f'**Question {i+1} of {len(QUESTIONS)}**')
    st.markdown(f'##### {q["q"]}')
    answer = st.radio(
        label='',
        options=q['opts'],
        key=f'q{i}',
        index=None,
        label_visibility='collapsed'
    )
    answers.append(answer)
    if answer is None:
        all_answered = False
    st.markdown('---')

# Analyze button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze = st.button('🔍 Analyze My Skin Type', disabled=not all_answered)

if not all_answered:
    st.info('👆 Please answer all 8 questions to see your result.')

# Results
if analyze and all_answered:
    skin_type, percentages, confidence = classify_skin(answers)
    info = SKIN_INFO[skin_type]

    st.balloons()
    st.divider()
    st.markdown('## 📊 Your Results')

    # Main result
    st.markdown(f'''
    <div class="result-box" style="border-color:{info["color"]}; background:{info["bg"]}">
        <h2 style="color:{info["color"]}; margin:0">{info["emoji"]} {skin_type} Skin</h2>
        <p style="font-size:1.1rem; margin:0.5rem 0 0 0; color:#555">{info["description"]}</p>
    </div>
    ''', unsafe_allow_html=True)

    # Score breakdown
    st.markdown('#### Score Breakdown')
    colors = {'Dry':'#3b82f6','Normal':'#10b981','Oily':'#f59e0b','Combination':'#8b5cf6'}
    for stype, pct in sorted(percentages.items(), key=lambda x: -x[1]):
        col_a, col_b = st.columns([3,1])
        with col_a:
            st.markdown(f'**{stype}**')
            st.markdown(
                f'<div class="score-bar" style="width:{pct}%; background:{colors[stype]}"></div>',
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(f'**{pct}%**')

    st.divider()

    # Two columns: characteristics + routine
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('#### 🔎 Your Skin Characteristics')
        for c in info['characteristics']:
            st.markdown(f'- {c}')

    with col2:
        st.markdown('#### 💆 Recommended Routine')
        for r in info['routine']:
            st.markdown(f'- {r}')

    st.divider()
    st.markdown('#### ⚠️ What To Avoid')
    st.warning(info['avoid'])

    st.divider()
    st.caption('⚕️ This tool is for educational purposes. For medical skin concerns, consult a dermatologist.')

    # Retake button
    if st.button('🔄 Retake Quiz'):
        st.rerun()
