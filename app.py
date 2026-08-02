"""
Streamlit demo app — Movie Review Sentiment Classifier
========================================================
Wired to the project's own Predictor (src/model/predictor.py), which loads
the fine-tuned model from A-Asif/movie-review-sentiment-distilbert-FT on the
Hugging Face Hub via ModelLoader (src/utils/model_loader.py).

Two modes, via tabs:
  1) Single Review   — paste a review, get a verdict + confidence bar
  2) Batch Evaluation — upload a labeled CSV, get accuracy/precision/recall/F1
                        + confusion matrix (same logic as evaluate.py, but
                        rendered inline in the app)

Run from the repo root (same level as main.py):
    streamlit run app.py
"""

import time

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

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

.section-label {
    font-family: 'Inter', sans-serif; font-size: 0.78rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}

.footer-note {
    text-align: center; font-family: 'Inter', sans-serif; font-size: 0.78rem; color: var(--muted);
    margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--rule);
}

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    background: var(--card); border: 1px solid var(--rule); border-radius: 6px 6px 0 0;
    font-family: 'Inter', sans-serif; color: var(--muted); padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-color: var(--gold) !important; }
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

# ---------------------------------------------------------------------------
# LOAD PREDICTOR (cached so the model only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor():
    return Predictor()


LABEL_MAP = {
    "positive": 1, "pos": 1, "label_1": 1, "1": 1,
    "negative": 0, "neg": 0, "label_0": 0, "0": 0,
}


def normalize_label(value):
    key = str(value).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unrecognized label value: {value!r}")
    return LABEL_MAP[key]


tab_single, tab_batch = st.tabs(["🎬 Single Review", "📊 Batch Evaluation"])

# ===========================================================================
# TAB 1 — SINGLE REVIEW PREDICTION
# ===========================================================================
with tab_single:
    st.markdown('<div class="ticket-divider"><span>submit a review</span></div>', unsafe_allow_html=True)

    EXAMPLES = [
        "An absolute masterpiece — the direction, the score, the performances, everything clicked.",
        "I wanted to walk out after twenty minutes. Wooden dialogue and a plot that goes nowhere.",
        "Solid popcorn entertainment. Not groundbreaking, but I had a genuinely fun time.",
    ]

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown(
        '<div class="section-label">Or try an example</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
        with col:
            if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                # Must write to the SAME key the text_area below uses, then force
                # a rerun. Streamlit binds session_state[key] to that widget once
                # it exists -- writing to a different variable gets silently
                # overridden by the widget's own state on the next run.
                st.session_state["review_text"] = ex
                st.rerun()

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

                # --- Confidence bar chart (Positive vs Negative) ---
                pos_conf = confidence if is_positive else 1 - confidence
                neg_conf = 1 - pos_conf

                fig, ax = plt.subplots(figsize=(5, 2.2))
                fig.patch.set_facecolor("#15120F")
                ax.set_facecolor("#15120F")
                bars = ax.barh(["Negative", "Positive"], [neg_conf, pos_conf],
                                color=["#B24C4C", "#5B8C6E"])
                ax.set_xlim(0, 1)
                ax.tick_params(colors="#A79C87")
                for spine in ax.spines.values():
                    spine.set_color("#3A342B")
                for bar, value in zip(bars, [neg_conf, pos_conf]):
                    ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                            f"{value:.1%}", va="center", color="#F1EAD9")
                st.pyplot(fig)

                # --- Session history ---
                st.session_state.history.append({"review": review_text[:60] + "...", "label": verdict_word, "confidence": confidence})

                if len(st.session_state.history) > 1:
                    st.markdown('<div class="section-label">Session History</div>', unsafe_allow_html=True)
                    hist_df = pd.DataFrame(st.session_state.history)
                    hist_df["confidence"] = hist_df["confidence"].round(3)
                    st.line_chart(hist_df["confidence"])
                    st.dataframe(hist_df, use_container_width=True)

            except Exception as e:
                st.error(
                    "Couldn't get a prediction. Make sure your model repo "
                    "(A-Asif/movie-review-sentiment-distilbert-FT) is public on the Hub, "
                    f"and that the app is run from the repo root.\n\nDetails: {e}"
                )

# ===========================================================================
# TAB 2 — BATCH EVALUATION (logic from evaluate.py, rendered inline)
# ===========================================================================
with tab_batch:
    st.markdown('<div class="ticket-divider"><span>evaluate on labeled data</span></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Upload a labeled CSV (e.g. data/test.csv)</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    col_a, col_b = st.columns(2)
    with col_a:
        text_col = st.text_input("Text column name", value="review")
    with col_b:
        label_col = st.text_input("Label column name", value="sentiment")

    run_eval = st.button("Run Evaluation", use_container_width=True)

    if run_eval:
        if uploaded_file is None:
            st.warning("Upload a CSV file first.")
        else:
            try:
                df = pd.read_csv(uploaded_file)

                if text_col not in df.columns or label_col not in df.columns:
                    st.error(
                        f"Columns '{text_col}' / '{label_col}' not found. "
                        f"Available columns: {list(df.columns)}"
                    )
                else:
                    predictor = load_predictor()

                    y_true, y_pred = [], []
                    progress = st.progress(0, text="Evaluating reviews...")

                    for i, row in df.iterrows():
                        text = str(row[text_col])
                        true_label = normalize_label(row[label_col])

                        label, _ = predictor.get_predicted_sentiment_with_confidence(text)
                        pred_label = normalize_label(label)

                        y_true.append(true_label)
                        y_pred.append(pred_label)

                        progress.progress((i + 1) / len(df), text=f"Evaluating reviews... {i + 1}/{len(df)}")

                    progress.empty()

                    acc = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average="binary", zero_division=0
                    )

                    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.2%}")
                    m2.metric("Precision", f"{precision:.2%}")
                    m3.metric("Recall", f"{recall:.2%}")
                    m4.metric("F1-score", f"{f1:.2%}")

                    with st.expander("Full classification report"):
                        st.text(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

                    # --- Confusion matrix ---
                    st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(4.5, 3.8))
                    fig.patch.set_facecolor("#15120F")
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax,
                        xticklabels=["Negative", "Positive"],
                        yticklabels=["Negative", "Positive"],
                        cbar=False,
                    )
                    ax.set_xlabel("Predicted", color="#A79C87")
                    ax.set_ylabel("Actual", color="#A79C87")
                    ax.tick_params(colors="#A79C87")
                    st.pyplot(fig)

                    # --- Downloadable results ---
                    results_df = df.copy()
                    results_df["predicted_label"] = ["Positive" if p == 1 else "Negative" for p in y_pred]
                    results_df["true_label_normalized"] = ["Positive" if t == 1 else "Negative" for t in y_true]
                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download detailed results CSV",
                        data=csv_bytes,
                        file_name="evaluation_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer-note">
    Fine-tuned on IMDB movie reviews &nbsp;·&nbsp; distilbert &nbsp;·&nbsp; Internship Project
</div>
""", unsafe_allow_html=True)