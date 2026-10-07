import streamlit as st
from src.predict import predict_spam


def set_custom_css():
    custom_css = """
    <style>
    /* Full Page Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #071A2B 0%,
            #0B2A4A 45%,
            #123B68 70%,
            #241B4B 100%
        ) !important;
        background-attachment: fixed !important;
    }
    
    /* General text color */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp label {
        color: #f8fafc !important;
    }
    
    /* Reduce top empty space */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
    }
    
    /* Hide Header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Desktop Layout Overrides */
    @media (min-width: 992px) {
        /* Left Column - Sticky */
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) {
            padding-right: 1.5rem;
            position: -webkit-sticky !important;
            position: sticky !important;
            top: 1.5rem !important;
            align-self: flex-start !important;
            height: fit-content !important;
        }
    }
    
    /* Right column Glassmorphism Panel */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    
    /* Reset inner columns inside the glass panel so they don't get the glass effect */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
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
    
    /* Text Area Label */
    .stTextArea label p {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Check Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC) !important;
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
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    }
    div.stButton > button:focus, div.stButton > button:active, div.stButton > button:focus-visible {
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
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Spam SMS Detection — CodSoft Task 4",
        page_icon="📩",
        layout="wide"
    )
    
    set_custom_css()
    
    left_col, right_col = st.columns([1, 1.3], gap="large")
    
    with left_col:
        # Project Title
        st.markdown('''
            <div style="margin-bottom: 2rem;">
                <p style="font-size: 0.85rem; font-weight: 600; color: #60a5fa; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">
                    CodSoft ML Internship — Task 4
                </p>
                <h1 style="font-size: 2rem; font-weight: 800; line-height: 1.2; margin: 0; background: linear-gradient(90deg, #ffffff 0%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Spam SMS Detection
                </h1>
            </div>
        ''', unsafe_allow_html=True)
        
        # Project Info Cards
        st.markdown('''
            <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                <div style="background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                    📩
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">About</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Classify SMS messages as Spam or Ham (not spam) using traditional Machine Learning techniques.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                <div style="background-color: rgba(34, 197, 94, 0.2); color: #4ade80; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                    🤖
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">ML Approach</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">TF-IDF text vectorization with Multinomial Naive Bayes, Logistic Regression, and Linear SVM classifiers compared and tuned.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                <div style="background-color: rgba(168, 85, 247, 0.2); color: #c084fc; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                    📊
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">Dataset</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">SMS Spam Collection — 5,169 unique messages (after duplicate removal) with ham/spam labels.</p>
                </div>
            </div>
            
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 1.2rem 1rem; margin-top: 1rem; display: flex; justify-content: space-around; align-items: center;">
                <div style="text-align: center;">
                    <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">5,169</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Messages</p>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div style="text-align: center;">
                    <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">2</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Classes</p>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div style="text-align: center;">
                    <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">TF-IDF</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Text Features</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
    with right_col:
        st.markdown('''
            <div style="display: flex; align-items: center; margin-bottom: 2rem;">
                <div style="background: #1e3a5f; min-width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
                </div>
                <div>
                    <h3 style="margin: 0; padding: 0; font-weight: 600; font-size: 1.3rem;">Check SMS Message</h3>
                    <p style="margin: 0; padding: 0; color: #cbd5e1; font-size: 0.9rem;">Enter an SMS message below to check if it is spam.</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        sms_input = st.text_area(
            "SMS Message",
            height=150,
            placeholder="Type or paste your SMS message here...",
            label_visibility="collapsed"
        )
        
        check_btn = st.button("📩 CHECK MESSAGE", use_container_width=True)
        
        if check_btn:
            # Input validation
            if not sms_input or sms_input.strip() == '':
                st.markdown('''
                    <div style="background: rgba(234, 179, 8, 0.15); border: 1px solid rgba(234, 179, 8, 0.4); border-radius: 12px; padding: 1.2rem; margin-top: 1.5rem; display: flex; align-items: center;">
                        <div style="background: #eab308; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                            ⚠️
                        </div>
                        <div>
                            <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Validation Error</p>
                            <h3 style="margin:0; color:#fff; font-size:1.1rem; font-weight: 700;">Please enter an SMS message before checking.</h3>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                label, probability = predict_spam(sms_input)
                
                if label == 'spam':
                    prob_display = f"{probability * 100:.1f}%" if probability is not None else ""
                    st.markdown(f'''
                        <div style="background: rgba(127, 29, 29, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center;">
                                <div style="background: #ef4444; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                                    🚫
                                </div>
                                <div>
                                    <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Classification Result:</p>
                                    <h3 style="margin:0; color:#fff; font-size:1.2rem; font-weight: 700;">SPAM MESSAGE</h3>
                                </div>
                            </div>
                            {f"""<div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 15px;">
                                <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Spam Probability:</p>
                                <h2 style="margin:0; color:#ffffff; font-size:1.8rem; font-weight: 700;">{prob_display}</h2>
                            </div>""" if probability is not None else ""}
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    prob_display = f"{(1 - probability) * 100:.1f}%" if probability is not None else ""
                    st.markdown(f'''
                        <div style="background: rgba(20, 83, 45, 0.3); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center;">
                                <div style="background: #22c55e; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                                    ✅
                                </div>
                                <div>
                                    <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Classification Result:</p>
                                    <h3 style="margin:0; color:#fff; font-size:1.2rem; font-weight: 700;">NOT SPAM</h3>
                                </div>
                            </div>
                            {f"""<div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 15px;">
                                <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Ham Confidence:</p>
                                <h2 style="margin:0; color:#ffffff; font-size:1.8rem; font-weight: 700;">{prob_display}</h2>
                            </div>""" if probability is not None else ""}
                        </div>
                    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
