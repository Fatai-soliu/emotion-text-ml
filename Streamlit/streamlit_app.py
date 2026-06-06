import streamlit as st
import pickle
import joblib
import numpy as np

st.set_page_config(layout="wide")

with st.sidebar:
    st.title(" ")

# ── Page config ────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='margin-bottom:-25px;'>
    🎭 EmotionLens
</h1>
<p style='margin-top:0;'>
    Emotion detection NLP
</p>
""", unsafe_allow_html=True)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit top padding */
.block-container { padding-top: 2rem; }

/* Emotion badge */
.emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.badge-sadness  { background: #E6F1FB; color: #0C447C; }
.badge-joy      { background: #FAEEDA; color: #633806; }
.badge-anger    { background: #FCEBEB; color: #791F1F; }
.badge-optimism { background: #EAF3DE; color: #27500A; }

/* Signal phrase pills */
.signal-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.signal-pill {
    background: #EEEDFE;
    color: #3C3489;
    border: 1px solid #AFA9EC;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 13px;
    font-family: monospace;
}

/* Info box */
.info-box {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #555;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Example texts ──────────────────────────────────────────────────────────────
EXAMPLES = {
    "😢 Sadness":   "I miss the way things used to be, nothing feels the same anymore.",
    "😊 Joy":       "I just got the news — I'm so happy I could burst, best day ever!",
    "😤 Anger":     "This is completely unacceptable. I'm sick of being ignored every single time.",
    "🌤 Optimism":  "Things may be hard right now, but I know something better is coming.",
}

# --- Loading model -------------------
@st.cache_resource
def load_model():
    return joblib.load("/Users/soliufatai/Documents/PersonalDocuments/Data_Science_ML_AI_Krish_Naik/Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main/2-Introduction/Intro/emotion-text-ml/models/svm_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("/Users/soliufatai/Documents/PersonalDocuments/Data_Science_ML_AI_Krish_Naik/Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main/2-Introduction/Intro/emotion-text-ml/models/tfidf_vectorizer.pkl")

model      = load_model()
vectorizer = load_vectorizer()

# --- Fix label classes -------

LABEL_MAP = {0: "Anger", 1: "Joy", 2: "Optimism", 3: "Sadness"}

# ── Model Pipeline ──────────────────────────

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(text: str) -> dict:
    # Transform text → numbers → prediction
    transformed = vectorizer.transform([text])
    emotion     = model.predict(transformed)[0]
    emotion = LABEL_MAP[emotion]
    scores      = model.decision_function(transformed)[0]
    proba       = softmax(scores)

    # Map labels to confidence scores
    classes    = model.classes_
    confidence = {LABEL_MAP[label]: round(float(prob), 3) for label, prob in zip(classes, proba)}

    # Extract signal phrases from top TF-IDF feature weights
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = transformed.toarray()[0]
    top_indices   = tfidf_scores.argsort()[::-1][:5]
    signals       = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

    if not signals:
        signals = ["(no strong signals detected)"]

    return {"emotion": emotion, "confidence": confidence, "signals": signals}

# ── Session state ──────────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── Example chips ──────────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
chip_cols = st.columns(len(EXAMPLES))
for col, (label, sample) in zip(chip_cols, EXAMPLES.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.input_text = sample

# ── Text input ─────────────────────────────────────────────────────────────────
text = st.text_area(
    "Enter text to analyse",
    value=st.session_state.input_text,
    placeholder="Type a sentence, tweet, journal entry, or message…",
    height=110,
    label_visibility="collapsed",
)

analyse = st.button("🧠 Analyse emotion", type="primary", use_container_width=False)

# ── Results ────────────────────────────────────────────────────────────────────
EMOTION_EMOJI = {"Sadness": "😢", "Joy": "😊", "Anger": "😤", "Optimism": "🌤"}
EMOTION_COLOR = {"Sadness": "#378ADD", "Joy": "#EF9F27", "Anger": "#E24B4A", "Optimism": "#639922"}
BADGE_CLASS   = {"Sadness": "badge-sadness", "Joy": "badge-joy",
                 "Anger": "badge-anger", "Optimism": "badge-optimism"}

if analyse and text.strip():
    result = predict(text)
    emotion     = result["emotion"]
    confidence  = result["confidence"]
    signals     = result["signals"]
    top_conf    = confidence[emotion]
    word_count  = len(text.split())

    st.divider()

    # Stat metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Words", word_count)
    m2.metric("Predicted emotion", emotion)
    m3.metric("Confidence", f"{top_conf:.0%}")

    st.divider()

    # Emotion badge
    badge_cls = BADGE_CLASS[emotion]
    emoji     = EMOTION_EMOJI[emotion]
    st.markdown(
        f'<div class="emotion-badge {badge_cls}">{emoji} {emotion}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Linear SVM · TF-IDF features · {top_conf:.0%} confidence")

    st.markdown("#### Confidence breakdown")

    # Sorted bars (highest first)
    sorted_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    for emo, score in sorted_conf:
        col_name, col_bar, col_pct = st.columns([0.25, 0.60, 0.15])
        with col_name:
            st.markdown(f"<span style='font-size:14px'>{EMOTION_EMOJI[emo]} {emo}</span>",
                        unsafe_allow_html=True)
        with col_bar:
            st.progress(min(float(score), 1.0))
        with col_pct:
            st.markdown(f"<span style='font-size:13px; color:grey;'>{score:.0%}</span>",
                        unsafe_allow_html=True)

    st.divider()

    # Signal phrases
    st.markdown("#### Key signal phrases detected")
    pills_html = "<div class='signal-container'>" + \
        "".join(f"<span class='signal-pill'>{s}</span>" for s in signals) + \
        "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

    st.divider()

    # Export
    export_col, _ = st.columns([0.3, 0.7])
    with export_col:
        export_text = (
            f"EmotionLens Result\n"
            f"------------------\n"
            f"Text      : {text}\n"
            f"Emotion   : {emotion}\n"
            f"Confidence: {top_conf:.0%}\n"
            f"Signals   : {', '.join(signals)}\n"
        )
        st.download_button(
            label="⬇ Export result",
            data=export_text,
            file_name="emotionlens_result.txt",
            mime="text/plain",
        )

elif analyse and not text.strip():
    st.warning("Please enter some text before analysing.")

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ How it works"):
    st.markdown("""
Text is vectorised with **TF-IDF** and classified by a **Linear SVM** trained on Twitter/X
data across four emotion categories: **Anger · Joy · Optimism · Sadness**.

Signal phrases are extracted from the highest-weight TF-IDF features contributing to the
predicted class. Confidence scores are derived from the SVM decision function via Platt scaling.
    """)