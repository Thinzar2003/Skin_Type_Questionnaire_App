import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title='Skin AI Agent',
    page_icon='🧴',
    layout='centered'
)

# ── STYLE ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans&display=swap');

html, body {
    font-family: 'DM Sans', sans-serif;
}
.main { background-color:#fdf8f4; }

.stButton > button {
    background: #2d2d2d;
    color: white;
    border-radius: 8px;
    padding: 0.6rem;
    width: 100%;
}

.result-box {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── AI AGENT ──────────────────────────────────────────────────────
class SkinAIAgent:

    def analyze_answers(self, answers):
        score = np.array(answers)

        scores = {
            "Dry": score[0] + score[3] + score[6],
            "Oily": score[1] + score[2] + score[7],
            "Combination": score[4] + score[6] + score[1],
            "Normal": sum(score) / 2
        }

        return scores

    def analyze_image(self, image):
        img = np.array(image)
        brightness = np.mean(img)

        if brightness > 180:
            return "Oily"
        elif brightness < 80:
            return "Dry"
        else:
            return "Normal"

    def decide(self, answer_scores, image_result):
        final_scores = answer_scores.copy()

        if image_result:
            final_scores[image_result] += 3

        skin = max(final_scores, key=final_scores.get)
        total = sum(final_scores.values())
        confidence = round(final_scores[skin] / total * 100, 1)

        return skin, confidence, final_scores

    def recommend(self, skin):
        routines = {
            "Dry": [
                "Hydrating cleanser",
                "Hyaluronic acid serum",
                "Heavy moisturizer",
                "SPF 50"
            ],
            "Oily": [
                "Foam cleanser",
                "Niacinamide serum",
                "Oil-free moisturizer",
                "Mattifying sunscreen"
            ],
            "Combination": [
                "Gentle cleanser",
                "Light moisturizer",
                "Treat T-zone separately",
                "SPF daily"
            ],
            "Normal": [
                "Simple cleanser",
                "Light hydration",
                "SPF daily",
                "Weekly exfoliation"
            ]
        }

        return routines[skin]

    def explain(self, skin, confidence):
        return f"AI predicts your skin is {skin} with {confidence}% confidence."

# ── INIT AGENT ───────────────────────────────────────────────────
agent = SkinAIAgent()

# ── UI ───────────────────────────────────────────────────────────
st.title("🧴 Skin AI Agent")

# IMAGE UPLOAD
st.subheader("📸 Upload Face Image (Optional)")
img_file = st.file_uploader("Upload image", type=["jpg","png"])

image_result = None
if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)
    image_result = agent.analyze_image(img)

st.divider()

# QUESTIONS
st.subheader("📝 Answer Questions")

questions = [
    "Skin feels tight",
    "Face gets shiny",
    "Acne frequency",
    "Feels dry",
    "Large pores",
    "Sensitive skin",
    "Rough texture",
    "Oil on tissue"
]

options = ["Low", "Medium", "High"]

answers = []
for i, q in enumerate(questions):
    val = st.selectbox(q, options, key=i, index=1)  # default = Medium
    answers.append(options.index(val))

# ANALYZE BUTTON
if st.button("🔍 Analyze with AI Agent"):

    answer_scores = agent.analyze_answers(answers)
    skin, confidence, final_scores = agent.decide(answer_scores, image_result)

    st.divider()

    # RESULT BOX
    st.markdown(f"""
    <div class="result-box">
        <h2>{skin} Skin</h2>
        <p>{agent.explain(skin, confidence)}</p>
    </div>
    """, unsafe_allow_html=True)

    # SCORES
    st.subheader("📊 Confidence Breakdown")
    for k, v in final_scores.items():
        percent = int((v / sum(final_scores.values())) * 100)
        st.write(f"{k}: {percent}%")
        st.progress(percent)

    st.divider()

    # ROUTINE
    st.subheader("💆 Recommended Routine")
    for r in agent.recommend(skin):
        st.write(f"• {r}")

    st.success("✅ AI Analysis Complete")
