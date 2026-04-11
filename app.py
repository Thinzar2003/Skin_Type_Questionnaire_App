import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title='Skin Type Analyzer',
    page_icon='🧴',
    layout='centered'
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Jost:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: "Jost", sans-serif; }
h1,h2,h3 { font-family: "Cormorant Garamond", serif !important; }
.result-card {
    padding: 1.5rem; border-radius: 14px;
    margin: 0.8rem 0; border-left: 5px solid;
    background: white; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
.score-bar {
    height: 10px; border-radius: 5px; margin: 3px 0 10px 0;
}
.compare-box {
    background: #f8f9fa; border-radius: 12px;
    padding: 1.2rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stButton > button {
    background: #1a1a2e; color: white; border: none;
    border-radius: 8px; padding: 0.6rem 2rem;
    font-family: "Jost", sans-serif; font-size: 1rem;
    width: 100%; transition: all 0.2s;
}
.stButton > button:hover { background: #16213e; transform: translateY(-1px); }
</style>
''', unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────
COLORS = {
    'Dry': '#3b82f6', 'Normal': '#10b981',
    'Oily': '#f59e0b', 'Combination': '#8b5cf6'
}
BG = {
    'Dry': '#eff6ff', 'Normal': '#ecfdf5',
    'Oily': '#fffbeb', 'Combination': '#f5f3ff'
}
EMOJI = {'Dry': '💧', 'Normal': '✨', 'Oily': '💫', 'Combination': '⚡'}

SKIN_INFO = {
    'Dry': {
        'description': 'Your skin lacks moisture and natural oils, causing tightness, flakiness, and dullness.',
        'characteristics': [
            'Feels tight after washing',
            'Rough or flaky texture',
            'Small barely-visible pores',
            'Prone to irritation',
            'Fine lines more visible'
        ],
        'routine': [
            'Cream-based gentle cleanser',
            'Hydrating toner (hyaluronic acid)',
            'Rich moisturizer with ceramides',
            'SPF 30+ sunscreen daily',
            'Overnight hydrating mask 2x/week'
        ],
        'avoid': 'Harsh cleansers, alcohol-based products, hot showers, over-exfoliating'
    },
    'Normal': {
        'description': 'Your skin is well-balanced with regulated sebum, small pores, and a naturally radiant appearance.',
        'characteristics': [
            'Balanced oil and moisture',
            'Small barely-visible pores',
            'Smooth even texture',
            'Rarely sensitive',
            'Naturally radiant'
        ],
        'routine': [
            'Gentle gel or foam cleanser',
            'Lightweight toner',
            'Light moisturizer',
            'SPF 30+ sunscreen daily',
            'Weekly exfoliation'
        ],
        'avoid': 'Heavy products that clog pores, skipping sunscreen'
    },
    'Oily': {
        'description': 'Your skin produces excess sebum giving a shiny look. More prone to breakouts but ages more slowly.',
        'characteristics': [
            'Shiny or greasy appearance',
            'Enlarged visible pores',
            'Prone to blackheads',
            'Makeup slides off',
            'Thick skin texture'
        ],
        'routine': [
            'Foaming cleanser with salicylic acid',
            'BHA or astringent toner',
            'Oil-free lightweight moisturizer',
            'Mattifying SPF 30+ sunscreen',
            'Niacinamide serum to control oil'
        ],
        'avoid': 'Heavy creams, coconut oil, skipping moisturizer'
    },
    'Combination': {
        'description': 'Your skin is oily in the T-zone (forehead, nose, chin) and dry or normal on the cheeks.',
        'characteristics': [
            'Oily T-zone',
            'Dry or normal cheeks',
            'Enlarged pores on nose',
            'Occasional T-zone breakouts',
            'Different zones need different care'
        ],
        'routine': [
            'Gentle balancing cleanser',
            'Balancing toner',
            'Light moisturizer on T-zone, richer on cheeks',
            'Lightweight SPF 30+ sunscreen',
            'Zone-specific treatments'
        ],
        'avoid': 'Same heavy product all over, harsh alcohol toners'
    }
}

QUESTIONS = [
    {
        'q': 'How does your skin feel ~1 hour after washing (no products)?',
        'opts': [
            'Very tight and uncomfortable',
            'Slightly tight',
            'Comfortable and balanced',
            'Fine, no particular feeling'
        ]
    },
    {
        'q': 'By midday, how does your skin look?',
        'opts': [
            'Very shiny all over',
            'Shiny only on T-zone (forehead/nose/chin)',
            'Same as morning',
            'Feels drier and tighter'
        ]
    },
    {
        'q': 'How often do you experience breakouts?',
        'opts': [
            'Frequently, all over face',
            'Occasionally, mainly T-zone',
            'Rarely',
            'Almost never, but skin is flaky'
        ]
    },
    {
        'q': 'How does your skin feel without moisturizer?',
        'opts': [
            'Very dry, flaky or itchy',
            'Slightly dry in some areas',
            'Normal, no issues',
            'Gets oily quickly'
        ]
    },
    {
        'q': 'How do your pores look up close?',
        'opts': [
            'Large and visible especially on nose',
            'Visible only on T-zone',
            'Small and barely visible',
            'Very small, skin looks tight'
        ]
    },
    {
        'q': 'How does your skin react to new products?',
        'opts': [
            'Often irritated or red',
            'Sometimes breaks out',
            'Rarely reacts',
            'Absorbs quickly, needs more'
        ]
    },
    {
        'q': 'What best describes your skin texture?',
        'opts': [
            'Rough, flaky or tight',
            'Smooth some areas, oily others',
            'Smooth and balanced overall',
            'Consistently shiny and greasy'
        ]
    },
    {
        'q': 'Press a tissue to your face — what happens?',
        'opts': [
            'Lots of oil all over',
            'Oil mainly from T-zone',
            'Very little oil',
            'Almost nothing, skin is dry'
        ]
    },
]

MAPPINGS = [
    {
        'Very tight and uncomfortable': {'Dry': 3},
        'Slightly tight': {'Dry': 2, 'Normal': 1},
        'Comfortable and balanced': {'Normal': 3},
        'Fine, no particular feeling': {'Normal': 2, 'Oily': 1}
    },
    {
        'Very shiny all over': {'Oily': 3},
        'Shiny only on T-zone (forehead/nose/chin)': {'Combination': 3},
        'Same as morning': {'Normal': 3},
        'Feels drier and tighter': {'Dry': 3}
    },
    {
        'Frequently, all over face': {'Oily': 2},
        'Occasionally, mainly T-zone': {'Combination': 2},
        'Rarely': {'Normal': 2},
        'Almost never, but skin is flaky': {'Dry': 2}
    },
    {
        'Very dry, flaky or itchy': {'Dry': 3},
        'Slightly dry in some areas': {'Combination': 2, 'Dry': 1},
        'Normal, no issues': {'Normal': 3},
        'Gets oily quickly': {'Oily': 3}
    },
    {
        'Large and visible especially on nose': {'Oily': 2, 'Combination': 1},
        'Visible only on T-zone': {'Combination': 3},
        'Small and barely visible': {'Normal': 2, 'Dry': 1},
        'Very small, skin looks tight': {'Dry': 2}
    },
    {
        'Often irritated or red': {'Dry': 2},
        'Sometimes breaks out': {'Oily': 1, 'Combination': 1},
        'Rarely reacts': {'Normal': 2},
        'Absorbs quickly, needs more': {'Oily': 2}
    },
    {
        'Rough, flaky or tight': {'Dry': 3},
        'Smooth some areas, oily others': {'Combination': 3},
        'Smooth and balanced overall': {'Normal': 3},
        'Consistently shiny and greasy': {'Oily': 3}
    },
    {
        'Lots of oil all over': {'Oily': 3},
        'Oil mainly from T-zone': {'Combination': 3},
        'Very little oil': {'Normal': 2},
        'Almost nothing, skin is dry': {'Dry': 3}
    },
]


# ── Classification functions ──────────────────────────────────────────
def classify_by_questionnaire(answers):
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}
    for i, ans in enumerate(answers):
        for k, v in MAPPINGS[i].get(ans, {}).items():
            scores[k] += v
    total = sum(scores.values()) or 1
    pcts  = {k: round(v / total * 100, 1) for k, v in scores.items()}
    best  = max(scores, key=scores.get)
    conf  = round(scores[best] / total * 100, 1)
    return best, pcts, conf


def analyze_image(pil_img):
    img     = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    face_detected = len(faces) > 0
    if face_detected:
        x, y, w, h = faces[0]
        pad    = 10
        region = img_bgr[max(0, y+pad):y+h-pad, max(0, x+pad):x+w-pad]
        if region.size == 0:
            region = img_bgr
    else:
        region = img_bgr

    # Feature extraction
    hsv            = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    brightness     = float(np.mean(hsv[:, :, 2]))
    brightness_std = float(np.std(hsv[:, :, 2]))
    saturation     = float(np.mean(hsv[:, :, 1]))
    gray_region    = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    texture_score  = float(cv2.Laplacian(gray_region, cv2.CV_64F).var())

    h_reg, w_reg = gray_region.shape
    t_zone       = gray_region[0:h_reg//3, w_reg//4:3*w_reg//4]
    cheeks       = gray_region[h_reg//3:2*h_reg//3, 0:w_reg//4]
    tzone_bright = float(np.mean(t_zone)) if t_zone.size > 0 else brightness
    cheek_bright = float(np.mean(cheeks)) if cheeks.size > 0 else brightness
    zone_diff    = abs(tzone_bright - cheek_bright)

    # Scoring
    scores = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0}

    if brightness > 180:
        scores['Oily'] += 3
    elif brightness > 150:
        scores['Oily'] += 1
        scores['Normal'] += 1
    elif brightness > 120:
        scores['Normal'] += 2
    else:
        scores['Dry'] += 2

    if brightness_std > 40:
        scores['Combination'] += 2
    elif brightness_std > 25:
        scores['Dry'] += 1
        scores['Combination'] += 1
    else:
        scores['Normal'] += 1
        scores['Oily'] += 1

    if saturation > 100:
        scores['Oily'] += 2
    elif saturation > 60:
        scores['Normal'] += 2
    else:
        scores['Dry'] += 2

    if texture_score > 300:
        scores['Dry'] += 2
    elif texture_score > 150:
        scores['Combination'] += 1
        scores['Normal'] += 1
    else:
        scores['Oily'] += 2

    if zone_diff > 20:
        scores['Combination'] += 3
    elif zone_diff > 10:
        scores['Combination'] += 1
        scores['Normal'] += 1
    else:
        scores['Normal'] += 1
        scores['Oily'] += 1

    total = sum(scores.values()) or 1
    pcts  = {k: round(v / total * 100, 1) for k, v in scores.items()}
    best  = max(scores, key=scores.get)
    conf  = round(scores[best] / total * 100, 1)

    features = {
        'Brightness': round(brightness, 1),
        'Saturation': round(saturation, 1),
        'Texture Score': round(texture_score, 1),
        'Zone Difference': round(zone_diff, 1),
        'Face Detected': face_detected
    }
    return best, pcts, conf, features


def show_result(skin_type, percentages, confidence, method):
    info  = SKIN_INFO[skin_type]
    color = COLORS[skin_type]
    bg    = BG[skin_type]
    emoji = EMOJI[skin_type]

    st.markdown(f'''
    <div class="result-card" style="border-color:{color};background:{bg}">
        <div style="font-size:0.8rem;color:#888;margin-bottom:4px">{method}</div>
        <h2 style="color:{color};margin:0">{emoji} {skin_type} Skin</h2>
        <p style="color:#555;margin:0.4rem 0 0">{info["description"]}</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('**Score Breakdown**')
    for stype, pct in sorted(percentages.items(), key=lambda x: -x[1]):
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.markdown(f'{stype}')
            st.markdown(
                f'<div class="score-bar" style="width:{pct}%;background:{COLORS[stype]}"></div>',
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(f'**{pct}%**')

    with st.expander('📋 Characteristics & Routine'):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Characteristics**')
            for c in info['characteristics']:
                st.markdown(f'- {c}')
        with c2:
            st.markdown('**Recommended Routine**')
            for r in info['routine']:
                st.markdown(f'- {r}')
        st.warning(f'**Avoid:** {info["avoid"]}')


# ── Main UI ───────────────────────────────────────────────────────────
st.title('🧴 Skin Type Analyzer')
st.markdown('*Discover your skin type using two methods — questionnaire and image analysis.*')
st.divider()

tab1, tab2, tab3 = st.tabs(['📝 Questionnaire', '📸 Image Analysis', '⚖️ Compare Both'])


# ── TAB 1: Questionnaire ──────────────────────────────────────────────
with tab1:
    st.markdown('### Answer 8 questions about your skin')
    st.markdown('*Based on dermatologist-validated criteria*')
    st.divider()

    answers      = []
    all_answered = True

    for i, q in enumerate(QUESTIONS):
        st.markdown(f'**Q{i+1}.** {q["q"]}')
        ans = st.radio(
            '', q['opts'], key=f'q{i}',
            index=None, label_visibility='collapsed'
        )
        answers.append(ans)
        if ans is None:
            all_answered = False
        st.markdown('')

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        btn1 = st.button('🔍 Analyze My Skin', key='btn_q', disabled=not all_answered)

    if not all_answered:
        st.info('👆 Please answer all 8 questions.')

    if btn1 and all_answered:
        skin_q, pcts_q, conf_q = classify_by_questionnaire(answers)
        st.session_state['q_result'] = (skin_q, pcts_q, conf_q)
        st.balloons()
        st.divider()
        st.markdown('## ✅ Your Result')
        show_result(skin_q, pcts_q, conf_q, 'Questionnaire Method')
        st.success('💡 Go to **⚖️ Compare Both** tab if you also did the image analysis!')


# ── TAB 2: Image Analysis ─────────────────────────────────────────────
with tab2:
    st.markdown('### Upload a clear face photo')
    st.markdown('*Best results: good lighting, face centered, no heavy makeup*')
    st.divider()

    uploaded = st.file_uploader(
        'Choose a photo', type=['jpg', 'jpeg', 'png'],
        key='img_upload'
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert('RGB')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil_img, caption='Your uploaded photo', use_column_width=True)

        with col2:
            st.markdown('**📊 Image Features Detected**')
            with st.spinner('Analyzing your skin...'):
                skin_i, pcts_i, conf_i, features = analyze_image(pil_img)
                st.session_state['img_result'] = (skin_i, pcts_i, conf_i)

            face_icon = '✅' if features['Face Detected'] else '⚠️'
            st.markdown(f'{face_icon} **Face detected:** {features["Face Detected"]}')
            st.markdown(f'💡 **Brightness:** {features["Brightness"]} / 255')
            st.markdown(f'🎨 **Saturation:** {features["Saturation"]} / 255')
            st.markdown(f'🔍 **Texture score:** {features["Texture Score"]}')
            st.markdown(f'📍 **Zone difference:** {features["Zone Difference"]}')

            if not features['Face Detected']:
                st.warning('No face detected. Using full image. Results may be less accurate.')

        st.divider()
        st.markdown('## ✅ Image Analysis Result')
        show_result(skin_i, pcts_i, conf_i, 'Image Analysis Method')
        st.success('💡 Go to **⚖️ Compare Both** tab to compare with your questionnaire result!')


# ── TAB 3: Compare ────────────────────────────────────────────────────
with tab3:
    st.markdown('### Side-by-side comparison of both methods')
    st.divider()

    q_done   = 'q_result'   in st.session_state
    img_done = 'img_result' in st.session_state

    if not q_done and not img_done:
        st.info('Complete both the **Questionnaire** and **Image Analysis** tabs first.')
    elif not q_done:
        st.warning('⬅️ Please complete the **Questionnaire** tab first.')
    elif not img_done:
        st.warning('⬅️ Please complete the **Image Analysis** tab first.')
    else:
        skin_q, pcts_q, conf_q = st.session_state['q_result']
        skin_i, pcts_i, conf_i = st.session_state['img_result']
        agree = skin_q == skin_i

        if agree:
            st.success(f'✅ Both methods agree: **{skin_q} Skin** — High confidence result!')
        else:
            st.warning(f'⚠️ Methods disagree — Questionnaire: **{skin_q}** | Image: **{skin_i}**')
            st.markdown('*The questionnaire is generally more reliable as your primary result.*')

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### 📝 Questionnaire')
            st.markdown(f'''
            <div class="compare-box" style="border-top:4px solid {COLORS[skin_q]};background:{BG[skin_q]}">
                <div style="font-size:2.5rem">{EMOJI[skin_q]}</div>
                <h3 style="color:{COLORS[skin_q]};margin:0.3rem 0">{skin_q} Skin</h3>
                <p style="color:#666;font-size:0.9rem">Confidence: {conf_q}%</p>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('**Scores:**')
            for stype, pct in sorted(pcts_q.items(), key=lambda x: -x[1]):
                st.markdown(f'- {stype}: {pct}%')

        with col2:
            st.markdown('#### 📸 Image Analysis')
            st.markdown(f'''
            <div class="compare-box" style="border-top:4px solid {COLORS[skin_i]};background:{BG[skin_i]}">
                <div style="font-size:2.5rem">{EMOJI[skin_i]}</div>
                <h3 style="color:{COLORS[skin_i]};margin:0.3rem 0">{skin_i} Skin</h3>
                <p style="color:#666;font-size:0.9rem">Confidence: {conf_i}%</p>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('**Scores:**')
            for stype, pct in sorted(pcts_i.items(), key=lambda x: -x[1]):
                st.markdown(f'- {stype}: {pct}%')

        st.divider()
        final = skin_q
        info  = SKIN_INFO[final]
        st.markdown('## 🎯 Final Recommendation')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Your Skin Characteristics:**')
            for c in info['characteristics']:
                st.markdown(f'- {c}')
        with c2:
            st.markdown('**Your Skincare Routine:**')
            for r in info['routine']:
                st.markdown(f'- {r}')
        st.error(f'**Avoid:** {info["avoid"]}')
        st.divider()
        st.caption('⚕️ For medical skin concerns, consult a licensed dermatologist.')
