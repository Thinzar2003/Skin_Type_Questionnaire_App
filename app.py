import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title='Skin AI Agent', page_icon='🧴', layout='centered')

# ── STYLE ─────────────────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: 'DM Sans'; }
.main { background-color:#fdf8f4; }
.result-box {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── AI AGENT CORE ────────────────────────────────────────────────
class SkinAIAgent:

    def __init__(self):
        self.skin_types = ["Dry", "Normal", "Oily", "Combination"]

    def analyze_answers(self, answers):
        score = np.array(answers)

        dry = score[0] + score[3] + score[6]
        oily = score[1] + score[2] + score[7]
        combo = score[4] + score[6] + score[1]
        normal = sum(score) / 2

        scores = {
            "Dry": dry,
            "Oily": oily,
            "Combination": combo,
            "Normal": normal
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
        # Combine both signals
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
        return f"AI Agent predicts your skin is {skin} with {confidence}% confidence based on your answers and image."

# ── INIT AGENT ───────────────────────────────────────────────────
agent = SkinAIAgent()

# ── UI ───────────────────────────────────────────────────────────
st.title("🧴 Skin AI Agent")

# IMAGE INPUT
st.subheader("📸 Upload Face Image (optional)")
img_file = st.file_uploader("Upload image", type=["jpg","png"])

image_result = None
if img_file:
    img = Image.open(img_file)
    st.image(img, width=200)
    image_result = agent.analyze_image(img)

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
for i,q in enumerate(questions):
    val = st.selectbox(q, options, key=i)
    answers.append(options.index(val))

# RUN AI
if st.button("🔍 Analyze with AI Agent"):

    answer_scores = agent.analyze_answers(answers)
    skin, confidence, final_scores = agent.decide(answer_scores, image_result)

    st.divider()

    # RESULT
    st.markdown(f"""
    <div class="result-box">
    <h2>{skin} Skin</h2>
    <p>{agent.explain(skin, confidence)}</p>
    </div>
    """, unsafe_allow_html=True)

    # SCORES
    st.subheader("📊 Confidence")
    for k,v in final_scores.items():
        st.write(f"{k}: {round(v,1)}")
        st.progress(int(v*10))

    # ROUTINE
    st.subheader("💆 Recommended Routine")
    routine = agent.recommend(skin)
    for r in routine:
        st.write(f"• {r}")

    st.success("AI Agent Completed Analysis ✅")
