"""
Streamlit demo app — Movie Review Sentiment Classifier
========================================================
Wired to the project's own Predictor (src/model/predictor.py), which loads
the fine-tuned model from A-Asif/movie-review-sentiment-distilbert-FT on the
Hugging Face Hub via ModelLoader (src/utils/model_loader.py).

Run from the repo root (same level as main.py):
    streamlit run app.py
"""

import time
import streamlit as st

from src.model.predictor import Predictor

st.set_page_config(
    page_title="The Critic's Desk — Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# STYLING — cinematic "critic's desk" theme
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg:        #15120F;
    --card:      #1E1A16;
    --gold:      #C9A24B;
    --gold-dim:  #8A7134;
    --parchment: #F1EAD9;
    --muted:     #A79C87;
    --positive:  #5B8C6E;
    --negative:  #B24C4C;
    --rule:      #3A342B;
}

.stApp { background: var(--bg); color: var(--parchment); }
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 760px;}

.hero { text-align: center; margin-bottom: 0.25rem; }
.hero-eyebrow {
    font-family: 'Inter', sans-serif; font-size: 0.78rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--gold); margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.6rem;
    line-height: 1.05; color: var(--parchment); margin: 0;
}
.hero-sub {
    font-family: 'Inter', sans-serif; font-size: 0.95rem; color: var(--muted);
    margin-top: 0.7rem;
}

.ticket-divider {
    display: flex; align-items: center; gap: 0.6rem; margin: 2rem 0 1.6rem 0;
    color: var(--gold-dim);
}
.ticket-divider::before, .ticket-divider::after {
    content: ""; flex: 1; height: 1px;
    background: repeating-linear-gradient(90deg, var(--rule), var(--rule) 6px, transparent 6px, transparent 12px);
}
.ticket-divider span { font-family: 'Fraunces', serif; font-style: italic; font-size: 0.85rem; white-space: nowrap; }

.stTextArea textarea {
    background: var(--card) !important; color: var(--parchment) !important;
    border: 1px solid var(--rule) !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-size: 1rem !important; padding: 1rem !important;
}
.stTextArea textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 1px var(--gold) !important; }
.stTextArea label {
    font-family: 'Inter', sans-serif; color: var(--muted) !important; font-size: 0.85rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}

.stButton > button {
    background: var(--gold) !important; color: #1A1610 !important; border: none !important;
    border-radius: 4px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.85rem !important;
    padding: 0.65rem 1.6rem !important; transition: all 0.15s ease;
}
.stButton > button:hover { background: #DDB65C !important; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

div[data-testid="column"] .stButton > button {
    background: transparent !important; color: var(--muted) !important; border: 1px solid var(--rule) !important;
    font-size: 0.78rem !important; padding: 0.4rem 0.9rem !important; text-transform: none; letter-spacing: normal;
}
div[data-testid="column"] .stButton > button:hover { border-color: var(--gold) !important; color: var(--gold) !important; }

.verdict-card {
    background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
    padding: 1.8rem 2rem; margin-top: 1.5rem; text-align: center; animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.verdict-label {
    font-family: 'Inter', sans-serif; font-size: 0.75rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}
.verdict-word { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.4rem; margin: 0.1rem 0 0.9rem 0; }
.verdict-word.positive { color: var(--positive); }
.verdict-word.negative { color: var(--negative); }

.confidence-track { width: 100%; height: 8px; background: var(--rule); border-radius: 4px; overflow: hidden; margin: 0.4rem 0 0.6rem 0; }
.confidence-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
.confidence-fill.positive { background: var(--positive); }
.confidence-fill.negative { background: var(--negative); }
.confidence-caption { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: var(--muted); }

.footer-note {
    text-align: center; font-family: 'Inter', sans-serif; font-size: 0.78rem; color: var(--muted);
    margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--rule);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Now Screening &nbsp;·&nbsp; Automated Review</div>
    <h1 class="hero-title">The Critic's Desk</h1>
    <div class="hero-sub">A fine-tuned DistilBERT model reads your review and hands down a verdict.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ticket-divider"><span>submit a review</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LOAD PREDICTOR (cached so the model only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor():
    return Predictor()


# ---------------------------------------------------------------------------
# EXAMPLE CHIPS
# ---------------------------------------------------------------------------
EXAMPLES = [
    "An absolute masterpiece — the direction, the score, the performances, everything clicked.",
    "I wanted to walk out after twenty minutes. Wooden dialogue and a plot that goes nowhere.",
    "Solid popcorn entertainment. Not groundbreaking, but I had a genuinely fun time.",
]

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

st.markdown(
    '<div style="font-family:Inter,sans-serif;font-size:0.78rem;color:#A79C87;'
    'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">Or try an example</div>',
    unsafe_allow_html=True,
)
cols = st.columns(3)
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    with col:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            # Must write to the SAME key the text_area below uses, then force
            # a rerun. Streamlit binds session_state[key] to that widget once
            # it exists -- writing to a different variable (as before) gets
            # silently overridden by the widget's own state on the next run.
            st.session_state["review_text"] = ex
            st.rerun()

# ---------------------------------------------------------------------------
# INPUT + PREDICT
# ---------------------------------------------------------------------------
review_text = st.text_area(
    "Your review",
    height=160,
    placeholder="Paste or write a movie review here...",
    key="review_text",
)

predict_clicked = st.button("Deliver Verdict", use_container_width=True)

if predict_clicked:
    if not review_text.strip():
        st.warning("Write or paste a review first — the critic needs something to read.")
    else:
        try:
            with st.spinner("Reading between the lines..."):
                predictor = load_predictor()
                label, confidence = predictor.get_predicted_sentiment_with_confidence(review_text)
                time.sleep(0.15)

            # Normalize label text/casing since model.config.id2label values
            # can vary (e.g. "POSITIVE"/"NEGATIVE" vs "positive"/"negative")
            is_positive = label.strip().lower().startswith("pos")
            css_class = "positive" if is_positive else "negative"
            verdict_word = "Positive" if is_positive else "Negative"
            icon = "★" if is_positive else "☹"

            st.markdown(f"""
            <div class="verdict-card">
                <div class="verdict-label">The Verdict</div>
                <div class="verdict-word {css_class}">{icon} &nbsp;{verdict_word}</div>
                <div class="confidence-track">
                    <div class="confidence-fill {css_class}" style="width:{confidence*100:.1f}%;"></div>
                </div>
                <div class="confidence-caption">{confidence*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(
                "Couldn't get a prediction. Make sure your model repo "
                "(A-Asif/movie-review-sentiment-distilbert-FT) is public on the Hub, "
                f"and that the app is run from the repo root.\n\nDetails: {e}"
            )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer-note">
    Fine-tuned on IMDB movie reviews &nbsp;·&nbsp; distilbert &nbsp;·&nbsp; Internship Project
</div>
""", unsafe_allow_html=True)


