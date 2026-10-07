import streamlit as st
import pandas as pd
from src.predict import predict_churn

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
    
    /* Inputs Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* Input Labels */
    .stNumberInput label p, .stSelectbox label p {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Predict Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    }
    
    /* Dropdown popup styling */
    ul[data-baseweb="menu"] {
        background-color: #0f172a !important;
    }
    li[data-baseweb="option"] {
        color: white !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide", initial_sidebar_state="collapsed")
    set_custom_css()
    
    # 30:70 Layout
    left_col, right_col = st.columns([0.3, 0.7], gap="large")
    
    with left_col:
        st.markdown('''
            <p style="font-size: 0.75rem; font-weight: 600; color: #60a5fa; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.3rem;">CODSOFT ML INTERNSHIP — TASK 3</p><br>
            <div style="font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin-bottom: 0.5rem;">
                <span style="color: white;">Customer Churn</span><br>
                <span style="color: #60a5fa;">Prediction</span>
            </div>
            <div style="font-size: 1.05rem; color: #cbd5e1; margin-bottom: 1.2rem; font-weight: 500;">
                Predict whether a customer is likely to churn using Machine Learning.
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                <div style="background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                    📊
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">Data Analysis</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Analyze customer demographics, account details, and product information to identify patterns related to customer churn.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                <div style="background-color: rgba(34, 197, 94, 0.2); color: #4ade80; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                    ⚙️
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">Machine Learning</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Compare Logistic Regression, Random Forest, and Gradient Boosting models to predict customer churn.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                <div style="background-color: rgba(168, 85, 247, 0.2); color: #c084fc; min-width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-right: 0.8rem;">
                    🏆
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: white;">Best Model</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">Tuned Random Forest achieved the best F1-Score of 0.608 and was selected as the final prediction model.</p>
                </div>
            </div>
            
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 1.2rem 1rem; margin-top: 1rem; display: flex; justify-content: space-around; align-items: center;">
                <div style="text-align: center;">
                    <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">10,000</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Total Records</p>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div style="text-align: center;">
                    <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">10</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Features</p>
                </div>
                <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.1);"></div>
                <div style="text-align: center;">
                    <p style="font-size: 1.3rem; font-weight: bold; color: white; margin: 0;">1</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Target Variable</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
    with right_col:
        st.markdown('''
            <div style="display: flex; align-items: center; margin-bottom: 2rem;">
                <div style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; min-width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-right: 1rem;">
                    👤
                </div>
                <div>
                    <h3 style="margin: 0; padding: 0; font-weight: 600; font-size: 1.3rem;">Customer Information</h3>
                    <p style="margin: 0; padding: 0; color: #cbd5e1; font-size: 0.9rem;">Enter the customer details below to predict churn.</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=20, value=5)
            
        with c2:
            balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
            has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"], index=1)
            is_active_member = st.selectbox("Active Member", ["Yes", "No"], index=0)
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
            
        predict_btn = st.button("📊 PREDICT CHURN", use_container_width=True)
        
        if predict_btn:
            input_data = {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                "EstimatedSalary": estimated_salary
            }
            
            prediction, probability = predict_churn(input_data)
            
            if prediction == 1:
                st.markdown(f'''
                    <div style="background: rgba(127, 29, 29, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <div style="background: #ef4444; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                                ⚠️
                            </div>
                            <div>
                                <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Customer Churn:</p>
                                <h3 style="margin:0; color:#fff; font-size:1.2rem; font-weight: 700;">LIKELY TO CHURN</h3>
                            </div>
                        </div>
                        <div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 15px;">
                            <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Churn Probability:</p>
                            <h2 style="margin:0; color:#ffffff; font-size:1.8rem; font-weight: 700;">{probability * 100:.1f}%</h2>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div style="background: rgba(20, 83, 45, 0.3); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <div style="background: #22c55e; color: white; min-width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-right: 15px;">
                                ✓
                            </div>
                            <div>
                                <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Customer Churn:</p>
                                <h3 style="margin:0; color:#fff; font-size:1.2rem; font-weight: 700;">NOT LIKELY TO CHURN</h3>
                            </div>
                        </div>
                        <div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 15px;">
                            <p style="margin:0; font-size:0.85rem; color:#cbd5e1; font-weight: 500;">Churn Probability:</p>
                            <h2 style="margin:0; color:#ffffff; font-size:1.8rem; font-weight: 700;">{probability * 100:.1f}%</h2>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
