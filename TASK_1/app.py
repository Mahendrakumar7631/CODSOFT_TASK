"""
Movie Genre Classification — Streamlit Application
CodSoft ML Internship — Task 1

A clean, professional Streamlit UI for predicting movie genres
from title and plot description using a TF-IDF + ML classifier.
"""

import streamlit as st
from src.predict import load_model, predict_genre


# ─── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Movie Genre Classification — CodSoft Task 1",
    page_icon="🎬",
    layout="wide"
)


# ─── Custom CSS (Task 4 style — dark blue/purple gradient) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(160deg, #0a0e1a 0%, #0f172a 40%, #111827 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Reduce top empty space */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
    }

    /* Hide Header & Menu */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Desktop Layout — left column sticky */
    @media (min-width: 992px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) {
            padding-right: 1.5rem;
            position: -webkit-sticky !important;
            position: sticky !important;
            top: 1.5rem !important;
            align-self: flex-start !important;
            height: fit-content !important;
        }
    }

    /* Right column — Glassmorphism Panel */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }

    /* Reset inner columns inside the glass panel */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* Text Input Styling */
    .stTextInput input {
        background-color: rgba(10, 18, 36, 0.85) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    .stTextInput input::placeholder {
        color: #8899aa !important;
        opacity: 1 !important;
    }
    .stTextInput label p {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(10, 18, 36, 0.85) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    .stTextArea textarea::placeholder {
        color: #8899aa !important;
        opacity: 1 !important;
    }
    .stTextArea label p {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* Predict Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6d28d9, #4f46e5, #3b82f6) !important;
        color: white !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-top: 15px;
        margin-bottom: 10px;
        border-radius: 8px !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #7c3aed, #6366f1, #60a5fa) !important;
        box-shadow: 0 4px 20px rgba(109, 40, 217, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:focus,
    div.stButton > button:active,
    div.stButton > button:focus-visible {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* Mobile responsiveness */
    @media (max-width: 991px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) {
            margin-top: 1rem !important;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─── Genre Emoji Mapping ────────────────────────────────────
GENRE_ICONS = {
    'Action/Adventure': '⚔️',
    'Comedy': '😂',
    'Drama': '🎭',
    'Documentary': '📽️',
    'Horror/Mystery': '👻',
    'Sci-Fi/Fantasy': '🚀',
    'Family/Animation': '🧸',
    'Entertainment': '🎤',
    'Short Film': '🎬',
    'Adult': '🔞',
}


# ─── Load Model ─────────────────────────────────────────────
@st.cache_resource
def get_model():
    """Load the model bundle once and cache it."""
    try:
        return load_model()
    except Exception:
        return None


model_bundle = get_model()


# ─── Two-Column Layout ──────────────────────────────────────
left_col, right_col = st.columns([1, 1.3], gap="large")


# ─── LEFT COLUMN — Project Info ──────────────────────────────
with left_col:
    # Project Title
    st.markdown('''
        <div style="margin-bottom: 2rem;">
            <p style="font-size: 0.85rem; font-weight: 600; color: #a78bfa; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">
                CodSoft ML Internship — Task 1
            </p>
            <h1 style="font-size: 2rem; font-weight: 800; line-height: 1.2; margin: 0; background: linear-gradient(90deg, #ffffff 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Movie Genre Prediction
            </h1>
        </div>
    ''', unsafe_allow_html=True)

    # About
    st.markdown('''
        <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
            <div style="background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                🎬
            </div>
            <div>
                <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">About</h4>
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Predict the genre of a movie from its title and plot description using traditional Machine Learning techniques.</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # ML Approach
    st.markdown('''
        <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
            <div style="background-color: rgba(34, 197, 94, 0.2); color: #4ade80; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                🧠
            </div>
            <div>
                <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">ML Approach</h4>
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Dual TF-IDF (word + character n-grams) with Logistic Regression, Linear SVM, and Multinomial Naive Bayes classifiers compared.</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Dataset
    st.markdown('''
        <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
            <div style="background-color: rgba(168, 85, 247, 0.2); color: #c084fc; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                📊
            </div>
            <div>
                <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">Dataset</h4>
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Movie Genre Classification Dataset — 54,214 training and 54,200 test samples with 27 original genres mapped to 10 clean categories.</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Dataset Statistics
    st.markdown('''
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 1.2rem 1rem; margin-top: 1rem; display: flex; justify-content: space-around; align-items: center;">
            <div style="text-align: center;">
                <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">54,214</p>
                <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Train Samples</p>
            </div>
            <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
            <div style="text-align: center;">
                <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">10</p>
                <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Genres</p>
            </div>
            <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
            <div style="text-align: center;">
                <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">TF-IDF</p>
                <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Text Features</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)


# ─── RIGHT COLUMN — Prediction ──────────────────────────────
with right_col:
    # Section Header
    st.markdown('''
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <div style="background: linear-gradient(135deg, #6d28d9, #4f46e5); min-width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/></svg>
            </div>
            <div>
                <h3 style="margin: 0; padding: 0; font-weight: 600; font-size: 1.3rem;">Enter Movie Details</h3>
                <p style="margin: 0; padding: 0; color: #cbd5e1; font-size: 0.9rem;">Provide the title and plot description to predict genre.</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Input Fields
    title = st.text_input(
        "Movie Title",
        placeholder="e.g. The Matrix, Titanic, Toy Story...",
    )

    description = st.text_area(
        "Plot Description",
        height=150,
        placeholder="Enter the movie's plot summary or description here...",
    )

    # Predict Button
    predict_clicked = st.button("🎬 PREDICT GENRE", use_container_width=True)

    # ─── Prediction Logic ────────────────────────────────────
    if predict_clicked:
        # Validation
        if not title or not title.strip() or not description or not description.strip():
            st.markdown('''
                <div style="background: rgba(234, 179, 8, 0.15); border: 1px solid rgba(234, 179, 8, 0.4); border-radius: 12px; padding: 1.2rem; margin-top: 1.5rem; display: flex; align-items: center;">
                    <div style="background: #eab308; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                        ⚠️
                    </div>
                    <div>
                        <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Validation Error</p>
                        <h3 style="margin:0; color:#fff; font-size:1.1rem; font-weight: 700;">Please enter both a movie title and plot description.</h3>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        elif model_bundle is None:
            st.markdown('''
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1.2rem; margin-top: 1.5rem; display: flex; align-items: center;">
                    <div style="background: #ef4444; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                        ❌
                    </div>
                    <div>
                        <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Model Error</p>
                        <h3 style="margin:0; color:#fff; font-size:1.1rem; font-weight: 700;">Model not found. Please train the model first using: python src/train_model.py</h3>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        else:
            # Run prediction
            genre, confidence = predict_genre(title, description, model_bundle)
            icon = GENRE_ICONS.get(genre, '🎬')

            # Build confidence display
            conf_section = ""
            if confidence is not None:
                conf_pct = f"{confidence * 100:.1f}%"
                conf_section = f'''
                    <div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 15px;">
                        <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Confidence</p>
                        <h2 style="margin:0; color:#ffffff; font-size:1.8rem; font-weight: 700;">{conf_pct}</h2>
                    </div>
                '''

            st.markdown(f'''
                <div style="background: rgba(20, 83, 45, 0.3); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center;">
                        <div style="background: #22c55e; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                            {icon}
                        </div>
                        <div>
                            <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Predicted Genre</p>
                            <h3 style="margin:0; color:#fff; font-size:1.2rem; font-weight: 700;">{genre}</h3>
                        </div>
                    </div>
                    {conf_section}
                </div>
            ''', unsafe_allow_html=True)
